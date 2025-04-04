# --coding:utf-8--
import torch
from models.factory import build_net

def LoadLocalW(net):
	# 加载预训练权重
	ori_module=torch.load('./weights/best.pt')
	ori_module_dict=ori_module['model']
	tar_module_dict=net.state_dict()
	torch.save(tar_module_dict,'tar_module_dict_ori.pth')

	# 筛选出名称和结构相同的模块权重
	matched_dict = {
			name : weight for name , weight in ori_module_dict.items()
			if name in tar_module_dict and weight.shape == tar_module_dict[ name ].shape
	}
	torch.save(matched_dict,'matched_dict.pth')

	# 更新新模型的权重字典
	tar_module_dict.update( matched_dict )

	# 加载到新模型
	net.load_state_dict( tar_module_dict )
	tar_module_dict=net.state_dict()
	torch.save(tar_module_dict,'tar_module_dict.pth')
	return net

net=build_net('train',1, 'dark')
tar_module_dict = net.state_dict()
torch.save(tar_module_dict,'test.pth')
# LoadLocalW(net)
