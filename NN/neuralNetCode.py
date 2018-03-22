import torch
from torch.autograd import Variable
import torchvision
import torch.utils.data
import numpy as np

#################################################Load Data#####################################################

filename_original = "D:\\LightFieldData\\Refocused\\refocused.npy"
filename_refocused = "D:\\LightFieldData\\Refocused\\original.npy"

original_data = np.load(filename_original).astype(np.float32)
refocused_data = np.load(filename_refocused).astype(np.float32)

# Reshape data for pytorch
original_data = np.transpose(original_data, (0,3,1,2))
refocused_data = np.transpose(refocused_data, (0,3,1,2))

# Train/Test split 
split = 700
train_orig = original_data[:split]
train_refc = refocused_data[:split]
test_orig = original_data[split:]
test_refc = refocused_data[split:]

num_images, ch, height, width, = original_data.shape

print(original_data.shape)

################################################################################################################

# Dummy image to feed through to determine initial information
data = Variable(torch.zeros(1,3,height,width),requires_grad=False).cuda()

###############################################################################################################

model = torch.nn.Sequential(
			torch.nn.Conv2d(3, 64, 3, stride=1, padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(64, 32, 3, stride=1, padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(32, 16, 3, stride=1, padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(16, 8, 3, stride=1, padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(8, 3, 3, stride=1, padding=1),
		)

model.cuda()

print(data.size())

residual = model(data)

print(residual.size())

learning_rate = 0.001
loss_fn = torch.nn.MSELoss(size_average=True)
params = model.parameters()
optimizer = torch.optim.Adam(params, lr=learning_rate)
		

################################################################################################################

b = 8 # Number of images per round
num_epochs = 800
skip = 10

original_image = original_data[0:1]
refocused_image = refocused_data[0:1]
learned_images = []
residual_images = []

# Setup for visulaization
current = Variable(torch.from_numpy(original_image), requires_grad=False).cuda()

# Setup Dataloaders
train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_orig), torch.from_numpy(train_refc))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=b, shuffle=True)
test_data = torch.utils.data.TensorDataset(torch.from_numpy(test_orig), torch.from_numpy(test_refc))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=b, shuffle=True)

for epoch in range(num_epochs):

	print("Epoch", epoch)
	i = 0
	
	for batch_input, batch_output in train_loader:
		
		batch_input = Variable(batch_input,requires_grad=False).cuda()
		batch_output = Variable(batch_output,requires_grad=False).cuda()
		
		# Forward Pass
		residual = model(batch_input)
		loss = loss_fn(batch_input + residual, batch_output)
		
		# Training step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# Printing statement
		i += 1
		if i % skip == 0:
			print(i,loss.data[0])
		
	
	# Visualization of progress
	residual = model(current)
	result = residual + current
	residual_images.append(residual.data.cpu().numpy())
	learned_images.append(result.data.cpu().numpy())

	
print("Saving...")
from scipy.misc import imsave
# Rearrange back into regular image format
original_image = np.transpose(original_image,(0,2,3,1))
refocused_image = np.transpose(refocused_image,(0,2,3,1))
imsave("original.png", original_image[0])
imsave("refocused.png", refocused_image[0])

# Save final learned result
learned_image = learned_images[-1]
learned_image = np.transpose(learned_image,(0,2,3,1))
imsave("learned.png", np.clip(learned_image,0.0,255.0)[0])

# Save learning steps as mp4		
import matplotlib.animation as animation
import matplotlib.pyplot as plt
mp4Writer = animation.writers['ffmpeg']
the_writer = mp4Writer(fps=30, metadata=dict(artist='Me'))
fig = plt.figure()
ims = []

plt.axis('off')

for image in residual_images:
	image = np.transpose(image,(0,2,3,1))
	image = np.clip(image+0.5,0.0,1.0)[0]
	ims.append([plt.imshow(image,vmin=0.0,vmax=1.0,animated=True)])
	
#for image in learned_images:
#	image = np.transpose(image,(0,2,3,1))
#	image = np.clip((image/255.0),0.0,1.0)[0]
#	ims.append([plt.imshow(image,vmin=0.0,vmax=1.0,animated=True)])

torch.save(model, "trained_model.pt")

im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=True)
im_ani.save("view.mp4", writer=the_writer)
print("Done")
