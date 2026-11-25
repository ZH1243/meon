"""Stream-ordered completion of a down projection with no local assignments."""
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack


class PublishEmptyDown:
    @cute.kernel
    def kernel(self, address: cute.Tensor, groups: cutlass.Int32):
        target = cute.make_ptr(cutlass.Int32, address[0], cute.AddressSpace.gmem, assumed_align=4)
        cute.arch.store(target.llvm_ptr, groups, sem="release", scope="sys")

    @cute.jit
    def __call__(self, address: cute.Tensor, groups: cutlass.Int32, stream: cuda.CUstream):
        self.kernel(address, groups).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


_compiled = {}


def prepare_empty_down_publisher(address, groups, stream):
    """Compile before native communication is released. The callable accepts the launch stream."""
    argument = from_dlpack(address)
    cu_stream = cuda.CUstream(stream.cuda_stream)
    if address.device not in _compiled:
        _compiled[address.device] = cute.compile(PublishEmptyDown(), argument, cutlass.Int32(groups), cu_stream)
    compiled = _compiled[address.device]
    return lambda launch_stream: compiled(argument, cutlass.Int32(groups), cuda.CUstream(launch_stream.cuda_stream))
