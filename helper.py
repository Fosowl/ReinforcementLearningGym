
import matplotlib.pyplot as plt
import torch

class brainScan:
    def __init__(self, model):
        self.activation = {}
        self.layer_count = 4
        self.layers = [None] * self.layer_count
        model.l1.register_forward_hook(self.get_activation('l1'))
        model.l2.register_forward_hook(self.get_activation('l2'))
        model.l3.register_forward_hook(self.get_activation('l3'))
        model.l4.register_forward_hook(self.get_activation('l4'))
        self.layers[0] = model.l1.out_features
        self.layers[1] = model.l2.out_features
        self.layers[2] = model.l3.out_features
        self.layers[3] = model.l4.out_features
        _, self.ax = plt.subplots(self.layer_count, 1)
        for axes in self.ax:
            axes.axis('off')
            axes.set_xticks([])
            axes.set_yticks([])

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def show(self):
        # Get the activations from the hook
        hidden_activations = [None] * self.layer_count
        normalized_activations = [None] * self.layer_count
        activation_image = [None] * self.layer_count

        hidden_activations[0] = self.activation['l1'][0]
        hidden_activations[1] = self.activation['l2'][0]
        hidden_activations[2] = self.activation['l3'][0]
        hidden_activations[3] = self.activation['l4'][0]
        # Normalize activations to grayscale values (0-255)
        for i in range(0, self.layer_count):
            normalized_activations[i] = (hidden_activations[i] - hidden_activations[i].min()) / (
                hidden_activations[i].max() - hidden_activations[i].min()
            ) * 255
            # Convert normalized activations to an image
            activation_image[i] = normalized_activations[i].view(1, self.layers[i]).cpu().numpy()
            # Display the activation image using Matplotlib
            self.ax[i].imshow(activation_image[i], cmap='viridis')
        plt.pause(0.0001)

class plotter:
    def __init__(self) -> None:
        _, self.ax = plt.subplots()

    def plot_values(self, values, show_result=False):
        durations_t = torch.tensor(values, dtype=torch.float)
        self.ax.plot(durations_t.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated