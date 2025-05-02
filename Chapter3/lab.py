import torch
import sys
import Ex3

# Hàm kiểm tra kết quả
def check_row_matmul(a, b, out):
    # Tạo kết quả tham chiếu bằng nhân ma trận trên CPU
    ref_out = torch.matmul(a.cpu(), b.cpu())
    print(a.size())
    print(b.size())
    print((ref_out.size()))
    print(out.size())
    #sys.exit(0)
    
    # Chuyển out về CPU để so sánh
    out_cpu = out.cpu()
    
    # Tính sai số tuyệt đối tối đa
    max_abs_error = torch.max(torch.abs(ref_out - out_cpu))
    mean_abs_error = torch.mean(torch.abs(ref_out - out_cpu))
    
    # Kiểm tra tính chính xác
    is_correct = torch.allclose(ref_out, out_cpu, rtol=1e-5, atol=1e-8)
    
    print(f"Max absolute error: {max_abs_error.item()}")
    print(f"Mean absolute error: {mean_abs_error.item()}")
    print(f"Result is correct: {is_correct}")
    
    return is_correct, max_abs_error.item()

# Hàm main để kiểm tra
def test_row_matmul():
    # Tạo dữ liệu mẫu
    row, com, col = 1024, 512, 768
    a = torch.randn(row, com, device='cuda', requires_grad=False, dtype=torch.float64)
    b = torch.randn(com, col, device='cuda', requires_grad=False, dtype=torch.float64)
    #out = torch.zeros(row, col, device='cuda', requires_grad=False)
    
    # Gọi hàm row_matmul
    out = Ex3.row_matmul(a, b)
    
    # Kiểm tra kết quả
    is_correct, max_error = check_row_matmul(a, b, out)
    
    if not is_correct and max_error > 1e-5:
        print("Error: Results do not match the reference output.")
    else:
        print("Test passed successfully!")

if __name__ == "__main__":
    test_row_matmul()