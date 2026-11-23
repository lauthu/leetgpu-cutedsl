import cutlass
import cutlass.cute as cute

THREADS = 256
VALUES = 4


@cute.kernel
def _color_inversion(image: cute.Tensor, pixels: cute.Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    pixel = bid * THREADS + tid

    if pixel < pixels:
        tI = cute.local_tile(image, (VALUES,), (pixel,))
        tO = cute.local_tile(image, (VALUES,), (pixel,))
        v = cute.make_rmem_tensor((VALUES), image.element_type)
        v.store(tI.load())

        for i in cutlass.range_constexpr(3):
            v[i] = cutlass.Uint8(255) - v[i]
        tO.store(v.load())


@cute.jit
def solve(image: cute.Tensor, width: cute.Int32, height: cute.Int32):
    pixels = width * height

    _color_inversion(image, pixels).launch(
        grid=(cute.ceil_div(pixels, THREADS), 1, 1), block=(THREADS, 1, 1)
    )
