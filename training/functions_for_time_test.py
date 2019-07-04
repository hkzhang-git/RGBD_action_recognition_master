import time
from torch.autograd import Variable


def val(model, val_loader, args):
    model.eval()
    output_buffer = []
    previous_video_id = ''
    test_results = {'pre_labels':{}}
    i = 0
    for data, target, video_ids in val_loader:
        if args.use_cuda:
            data = data.cuda()
        data = Variable(data/127.5-1, volatile=True)
        output = model(data)

        for j in range(len(video_ids)):
            if not (i == 0 and j == 0) and video_ids[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id, test_results, args.topk)
                output_buffer = []
            output_buffer.append(output[j].data.cpu())
            previous_video_id = video_ids[j]
        i += 1
    calculate_video_results(output_buffer, previous_video_id, test_results, args.topk)
    return test_results





