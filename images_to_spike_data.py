import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils
from torch.utils.data import DataLoader
from snntorch import spikegen



# Training Parameters
batch_size=128
data_path='/data/mnist'
num_classes = 10  # MNIST has 10 output classes

# Torch Variables_
dtype = torch.float

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)


#Creating dataloader
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

num_steps = 100 #sets the duration of the spiketrain from the static input image

# Iterate through minibatches
data = iter(train_loader)
data_it, targets_it = next(data)

#RATE CODING
spike_data = spikegen.rate(data_it, num_steps=num_steps)
print(spike_data.size())






"""

#https://snntorch.readthedocs.io/en/latest/_modules/snntorch/spikegen.html

def rate(
    data, num_steps=False, gain=1, offset=0, first_spike_time=0, time_var_input=False
    ):
    
    if first_spike_time < 0 or num_steps < 0:
        raise Exception("``first_spike_time`` and ``num_steps`` cannot be negative.")

    if first_spike_time > (num_steps - 1):
        if num_steps:
            raise Exception(
                f"first_spike_time ({first_spike_time}) must be equal to or less than num_steps-1 ({num_steps-1})."
            )
        if not time_var_input:
            raise Exception(
                "If the input data is time-varying, set ``time_var_input=True``.\n If the input data is not time-varying, ensure ``num_steps > 0``."
            )

    if first_spike_time > 0 and not time_var_input and not num_steps:
        raise Exception(
            "``num_steps`` must be specified if both the input is not time-varying and ``first_spike_time`` is greater than 0."
        )

    if time_var_input and num_steps:
        raise Exception(
            "``num_steps`` should not be specified if input is time-varying, i.e., ``time_var_input=True``.\n The first dimension of the input data + ``first_spike_time`` will determine ``num_steps``."
        )

    device = data.device

    # intended for time-varying input data
    if time_var_input:
        spike_data = rate_conv(data)
        
        # zeros are added directly to the start of 0th (time) dimension
        if first_spike_time > 0:
            spike_data = torch.cat(
                (
                    torch.zeros(
                        tuple([first_spike_time] + list(spike_data[0].size())),
                        device=device,
                        dtype=dtype,
                    ),
                    spike_data,
                )
            )

    # intended for time-static input data
    else:

        # Generate a tuple: (num_steps, 1..., 1) where the number of 1's = number of dimensions in the original data.
        # Multiply by gain and add offset.
        time_data = (
            data.repeat(
                tuple([num_steps] + torch.ones(len(data.size()), dtype=int).tolist())
            )
            * gain
            + offset
        )

        spike_data = rate_conv(time_data)

        # zeros are multiplied by the start of the 0th (time) dimension
        if first_spike_time > 0:
            spike_data[0:first_spike_time] = 0

    return spike_data
    
    
def rate_conv(data):
     # Clip all features between 0 and 1 so they can be used as probabilities.
    clipped_data = torch.clamp(data, min=0, max=1)

    # pass time_data matrix into bernoulli function.
    spike_data = torch.bernoulli(clipped_data)

    return spike_data

    print(rate_conv(spike_data.size()))

    """

