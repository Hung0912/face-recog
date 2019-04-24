import  dlib
import dlib.cuda as cuda

print(dlib.DLIB_USE_CUDA)
print(cuda.get_num_devices())
