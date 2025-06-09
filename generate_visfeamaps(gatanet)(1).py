import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from config1 import test_data
from misc import check_mkdir
from VDTNet_visual import Mnet
from pylab import *
import os
import logging

torch.manual_seed(2018)
torch.cuda.set_device(1)


ckpt_path = './model/'
exp_name = ''

args = {
    'snapshot': 'VDTNet_epoch_best12.25_3',
    'crf_refine':False,
    'save_results': True
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#####
depth_transform = transforms.ToTensor()
thermal_transform = transforms.ToTensor()
#####
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

to_test = {'test':test_data}

def main():
    t0 = time.time()
    net = Mnet().cuda()
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot'] + '.pth')))
    net.eval()
    save_path = './weight/'
    logging.basicConfig(filename=save_path + 'w1w2.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                        datefmt='%Y-%m-%d %I:%M:%S %p')

    with torch.no_grad():
        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, ' %s_%s' % ( name, args['snapshot'])))
            root1 = os.path.join(root, 'RGB')
            img_list = [os.path.splitext(f)[0] for f in os.listdir(root1) if f.endswith('.png')]
            for idx, img_name in enumerate(img_list, start=0):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                img1 = Image.open(os.path.join(root, 'RGB',img_name + '.png')).convert('RGB')
                ########################################################
                depth = Image.open(os.path.join(root, 'T', img_name + '.png')).convert('L')
                thermal = Image.open(os.path.join(root, 'D', img_name + '.png')).convert('L')

                #######################################################
                img1 = img1.resize([352, 352])
                #######
                depth = depth.resize([352, 352])

                thermal = thermal.resize([352, 352])

                number = int(img_name)

                #######
                # img_var = Variable(img_transform(img1).unsqueeze(0), volatile=True).cuda()
                img_var = img_transform(img1).unsqueeze(0).cuda()
                #######
                # depth = Variable(depth_transform(depth).unsqueeze(0), volatile=True).cuda()
                depth = depth_transform(depth).unsqueeze(0).cuda()
                thermal = thermal_transform(thermal).unsqueeze(0).cuda()
                #######
                # (rd1_w1,rt1_w1,rd1_w2,rt1_w2,t1_w,d1_w,
                #  rd2_w1,rt2_w1,rd2_w2,rt2_w2,t2_w,d2_w,
                #  rd3_w1,rt3_w1,rd3_w2,rt3_w2,t3_w,d3_w,
                #  rd4_w1,rt4_w1,rd4_w2,rt4_w2,t4_w,d4_w,
                #  rd5_w1,rt5_w1,rd5_w2,rt5_w2,t5_w,d5_w)= net(img_var,thermal,depth)
                (vd1,vt1,vd2,vt2,vd3,vt3,vd4,vt4,vd5,vt5,f1,f2,f3,f4,f5) = net(img_var,thermal,depth)
                # logging.info(
                #     'image_number: [{:4d}],  t1_w: {:.4f} d1_w: {:.4f} t2_w: {:.4f}  d2_w:{:.4f}, t3_w: {:.4f} d3_w: {:.4f} t4_w: {:.4f} d4_w: {:.4f} t5_w: {:.4f} d5_w: {:.4f} '.
                #     format(number, t1_w.item(),d1_w.item(), t2_w.item(),d2_w.item(), t3_w.item(),d3_w.item(), t4_w.item(),d4_w.item(), t5_w.item(),d5_w.item()
                #            ))

                # a = [rd1_w1,rt1_w1,rd1_w2,rt1_w2,
                #      rd2_w1, rt2_w1, rd2_w2, rt2_w2,
                #      rd3_w1, rt3_w1, rd3_w2, rt3_w2,
                #      rd4_w1, rt4_w1, rd4_w2, rt4_w2,
                #      rd5_w1, rt5_w1, rd5_w2, rt5_w2
                #      ]
                a = [vd1,vt1,vd2,vt2,vd3,vt3,vd4,vt4,vd5,vt5,f1,f2,f3,f4,f5]
                for i in range(len(a)):
                    visualize_feature_map(a[i], img_name, i)

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch,img_name,num):
    # print(img_batch.size()[0:])

    feature_map = torch.squeeze(img_batch, 0).cuda()
    # print(feature_map.shape)
    if(len(feature_map.size())==2):
        feature_map = torch.unsqueeze(feature_map,0)


    # feature_map_combination = []
    # num_pic = feature_map.shape[0]
    # row, col = get_row_col(num_pic)
    #
    # for i in range(0, num_pic):
    #     feature_map_split = feature_map[i, :, :]
    #     feature_map_combination.append(feature_map_split)
    #
    # feature_map_sum = sum(ele for ele in feature_map_combination)

    feature_map_sum = torch.sum(feature_map,dim=0,keepdim=False)
    feature_map_sum = feature_map_sum.cuda().data.cpu()
    plt.imshow(feature_map_sum)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig('Feature_map/'+img_name+'_'+str(num)+".png", bbox_inches='tight', dpi=18, pad_inches=0.0)
    plt.savefig('Feature_map/' + img_name + '_' + str(num) + ".png", bbox_inches='tight', dpi=18, pad_inches=0.0)
    plt.close()
if __name__ == '__main__':
    main()
