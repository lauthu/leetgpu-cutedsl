"""Color inversion on 32-bit words: one XOR per pixel.

A pixel is four bytes, R G B A. Read as one little-endian 32-bit word, R, G
and B are its three low bytes and alpha is its high byte. For a byte,
255 - x is the same as x ^ 0xFF, so inverting the colors and keeping alpha is
one operation on the word: word ^ 0x00FFFFFF.

`solve` views the image as one word per pixel. Every thread owns PIXELS
consecutive pixels. A thread whose pixels are all inside the image loads them
as one vector, XORs and stores. The thread that owns the last, partial group
handles it one pixel at a time.

What changes relative to v1.py, which gave each thread one pixel and moved it
byte by byte through a register tensor: a thread now makes one load and one
store for four pixels, and there are a quarter as many threads.

The word view needs a promise about alignment. The image arrives as bytes,
with a pointer that promises nothing. Timed in one process on the platform's
H100 at 4096 x 5120, a word pointer that promises only 1 byte was three times
slower than one that promises 4, because each word is then put together from
single bytes. So `solve` checks the address and promises 4 bytes when that is
true. The other branch is the same kernel without the promise: correct at any
address, slower, and it costs one more compilation.
"""

import cutlass
import cutlass.cute as cute

THREADS = 256  # per block
PIXELS = 4  # per thread
WORD_BYTES = 4
RGB = 0x00FFFFFF  # the three low bytes of a pixel's word


def _words(image, pixels, align):
    """The image as a tensor of `pixels` 32-bit words, one per pixel.

    `align` is the alignment in bytes that the new pointer promises to the
    compiler. The address itself is not changed, so the caller must have
    checked it.
    """
    ptr = cute.make_ptr(
        cutlass.Uint32,
        image.iterator.toint(),
        image.iterator.memspace,
        assumed_align=align,
    )
    return cute.make_tensor(ptr, cute.make_layout(pixels, stride=1))


@cute.kernel
def _invert(words: cute.Tensor, pixels: cute.Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    # This thread owns group number `group`: pixels [first, first + PIXELS).
    group = bid * THREADS + tid
    first = group * PIXELS

    if first + PIXELS <= pixels:
        # The whole group is inside the image. Tiling by PIXELS gives it a
        # static shape, which load() and store() need.
        t = cute.local_tile(words, (PIXELS,), (group,))
        t.store(t.load() ^ cutlass.Uint32(RGB))
    else:
        # The last group when the pixel count is not a multiple of PIXELS,
        # and the threads past the end: one guarded pixel at a time.
        for k in cutlass.range_constexpr(PIXELS):
            i = first + k
            if i < pixels:
                words[i] = words[i] ^ cutlass.Uint32(RGB)


def _launch(image, pixels, align):
    _invert(_words(image, pixels, align), pixels).launch(
        grid=(cute.ceil_div(pixels, THREADS * PIXELS), 1, 1),
        block=(THREADS, 1, 1),
    )


@cute.jit
def solve(image: cute.Tensor, width: cute.Int32, height: cute.Int32):
    pixels = width * height
    if (image.iterator.toint() & (WORD_BYTES - 1)) == 0:
        _launch(image, pixels, WORD_BYTES)
    else:
        _launch(image, pixels, 1)
