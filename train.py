import os
import torch
import data
import loss
import utils
import time
from option import args
from model import ThinAge, TinyAge
from test import test
from datetime import datetime

models = {'ThinAge': ThinAge, 'TinyAge': TinyAge}

def get_group(age):
	if 0 <= age <= 5:
		return 0
	if 6 <= age <= 10:
		return 1
	if 11 <= age <= 20:
		return 2
	if 21 <= age <= 30:
		return 3
	if 31 <= age <= 40:
		return 4
	if 41 <= age <= 60:
		return 5
	if 61 <= age:
		return 6

def get_model(pretrained=False):
	model = args.model_name
	assert(model in models)
	if pretrained:
		path = os.path.join('./checkpoint/{}.pt'.format(model))
		assert os.path.exists(path)
		return torch.load(path)
	model = models[model]()

	return model


def main():
	model = get_model()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	loader = data.Data(args, "train").train_loader
	val_loader = data.Data(args, "valid").valid_loader
	rank = torch.Tensor([i for i in range(101)]).cuda()
	best_mae = 10000

	group = {0:"0-5", 1:"6-10", 2:"11-20", 3:"21-30", 4:"31-40", 5:"41-60", 6:"61-"}
	group_count = torch.zeros(7)
	to_count = True

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
			ages = torch.sum(outputs*rank, dim=1)
			loss1 = loss.kl_loss(outputs, label)
			loss2 = loss.L1_loss(ages, age)
			total_loss = loss1 + loss2
			total_loss.backward()
			optimizer.step()
			current_time = time.time()
			print('[Epoch:{}] \t[batch:{}]\t[loss={:.4f}]'.format(
				i, j, total_loss.item()), end = " ")
		torch.cuda.empty_cache()
		model.eval()
		count = 0
		error = 0
		total_loss = 0
		with torch.no_grad():
			for inputs in val_loader:
				img, label, age = inputs
				if to_count:
					for p in age:
						group_count[get_group(p.item())] += 1
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

				for ind, a in enumerate(ages): 
					if abs(age[ind].item() - a) < 1:
						correct_count[get_group(age[ind].item())] += 1
						correct_group[get_group(age[ind].item())] += 1
					elif get_group(age[ind].item()) == get_group(a):
						correct_group[get_group(age[ind].item())] += 1
		mae = error / count
		if to_count:
			for ind, p in enumerate(group_count):
				if p == 0:
					group_count[ind] = 1
			to_count = False
		print("Correct group rate:")
		print(correct_group/group_count)
		print("Correct age rate:")
		print(correct_count/group_count)
		rate = (correct_group, correct_count)

		if mae < best_mae:
			print("Epoch: {}\tVal loss: {:.5f}\tVal MAE: {:.4f} improved from {:.4f}".format(i, total_loss/count, mae, best_mae))
			best_mae = mae
			torch.save(model, "checkpoint/epoch{:03d}_{}_{:.5f}_{:.4f}_{}_{}_pretraining.pth".format(i, args.dataset, total_loss/count, best_mae, datetime.now().strftime("%Y%m%d"), args.model_name))
			best_rate = rate
		else:
			print("Epoch: {}\tVal loss: {:.5f}\tBest Val MAE: {:.4f} not improved, current MAE: {:.4f}".format(i, total_loss/count, best_mae, mae))
		torch.cuda.empty_cache()
	
	print("Finish, with best MAE")
	print("Correct group:")
	print(rate[0])
	print(rate[0]/group_count)
	print("Correct age:")
	print(rate[1])
	print(rate[1]/group_count)
	


if __name__ == '__main__':
	main()
