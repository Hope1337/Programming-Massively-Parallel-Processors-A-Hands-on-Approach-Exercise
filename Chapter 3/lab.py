import torch
import hihi

# Tạo 2 tensor trên GPU
a = torch.randn(5, device='cuda')
b = torch.randn(5, device='cuda')

# Gọi hàm vector_add của bạn
out = hihi.vector_add(a, b)

print(out)
print(torch.allclose(out, a + b))  # Kiểm tra xem kết quả có đúng không
