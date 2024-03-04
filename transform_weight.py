import torch
from ultralytics.models.yolo.detect.c2f_transfer import replace_c2f_with_c2f_v2, replace_c2f_v2_with_c2f

if __name__ == '__main__':
    input = torch.randn((1, 3, 640, 640))
    model = torch.load('test.pt')['model'].float().cpu()
    pre_res = model(input)[0]
    replace_c2f_v2_with_c2f(model)
    after_res = model(input)[0]
    print(torch.mean(pre_res - after_res))
    torch.save({'model':model.cuda().half()}, 'test_1.pt')