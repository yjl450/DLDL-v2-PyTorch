import os
import torch
import data
import loss
import utils
import time
from option import args
from model import ThinAge, TinyAge, get_model
from test import test
from datetime import datetime

models = {'ThinAge': ThinAge, 'TinyAge': TinyAge}


# def get_model(pretrained=False):
# 	model = args.model_name
# 	assert(model in models)
# 	if pretrained:
# 		path = os.path.join('./checkpoint/{}.pt'.format(model))
# 		assert os.path.exists(path)
# 		return torch.load(path)
# 	model = models[model]()

# 	return model


def main():
	model = get_model()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	loader = data.Data(args, "train").train_loader
	val_loader = data.Data(args, "valid").valid_loader
	rank = torch.Tensor([i for i in range(101)]).cuda()
	best_mae = 10000
	for i in range(args.epochs):
		lr = 0.001 if i < 30 else 0.0001
		optimizer = utils.make_optimizer(args, model, lr)
		model.train()
		print('Learning rate:{}'.format(lr))
		start_time = time.time()
		for j, inputs in enumerate(loader):
			img, label, age = inputs
			img = img.to(device)
			label = label.to(device)
			age = age.to(device)
			optimizer.zero_grad()
			outputs = model(img)
			outputs = torch.sigmoid(outputs)
			ages = torch.sum(outputs*rank, dim=1)
			loss1 = loss.kl_loss(outputs, label)
			loss2 = loss.L1_loss(ages, age)
			total_loss = loss1 + loss2
			total_loss.backward()
			optimizer.step()
			current_time = time.time()
			print('[Epoch:{}] \t[batch:{}]\t[loss={:.4f}]'.format(
				i, j, total_loss.item()))
		torch.cuda.empty_cache()
		model.eval()
		count = 0
		error = 0
		total_loss = 0
		with torch.no_grad():
			for inputs in val_loader:
				img, label, age = inputs
				count += len(age)
				img = img.to(device)
				label = label.to(device)
				age = age.to(device)
				outputs = model(img)
				ages = torch.sum(outputs*rank, dim=1)
				loss1 = loss.kl_loss(outputs, label)
				loss2 = loss.L1_loss(ages, age)
				total_loss += loss1 + loss2
				error += torch.sum(abs(ages - age))
		mae = error / count
		if mae < best_mae:
			print("Epoch: {}\tVal loss: {:.5f}\tVal MAE: {:.4f} improved from {:.4f}".format(i, total_loss/count, mae, best_mae))
			best_mae = mae
			torch.save(model, "checkpoint/epoch{:03d}_{}_{:.5f}_{:.4f}_{}_{}_pretraining.pth".format(i, args.dataset, total_loss/count, best_mae, datetime.now().strftime("%Y%m%d"), args.model_name))
		else:
			print("Epoch: {}\tVal loss: {:.5f}\tBest Val MAE: {:.4f} not improved, current MAE: {:.4f}".format(i, total_loss/count, best_mae, mae))
		torch.cuda.empty_cache()
		# torch.save(model.state_dict(),
		#            './pretrained/{}_dict.pt'.format(args.model_name))
		# print('Test: Epoch=[{}]'.format(i))
		# if (i+1) % 2 == 0:
		#     test()


if __name__ == '__main__':
	main()
