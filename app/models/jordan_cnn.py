from torch import Tensor, nn


class JordanCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.__parameters_validation(num_classes, dropout)
        self.features = nn.Sequential(
            self.__make_conv_block(3, 32),             # Initial feature extraction
            self.__make_conv_block(32, 64),            # Intermediate feature extraction
            self.__make_conv_block(64, 128),           # Higher-level feature extraction
            self.__make_conv_block(128, 256),          # High-level feature extraction
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))       # Reduce spatial dimensions to 1 × 1 for classification
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),                   # Flatten feature maps into a one-dimensional vector
            nn.Linear(256, 128),                       # Reduce features from 256 to 128
            nn.ReLU(),                                 # Apply ReLU activation
            nn.Dropout(p=dropout),                     # Randomly deactivate neurons to reduce overfitting
            nn.Linear(128, num_classes),               # Map features to class scores
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
    
    @staticmethod
    def __parameters_validation(num_classes: int, dropout: float) -> None:
        """
        Validates parameters for JordanCNN model.
        
        :param num_classes: total number of classification classes.
        :param dropout: rate of neuron dropout.
        :return:
        """
        # The num_classes must be a real number instead of boolean.
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes should be an integer.")
        # The num_classes must be larger than 2 to prevent meaningless classifier.
        if num_classes < 2:
            raise ValueError("num_classes should be greater than 2.")
        # The dropdown must be a real number or float.
        if isinstance(dropout, bool) or not isinstance(dropout, (float, int)):
            raise TypeError("dropout should be a float or integer.")
        # The dropdown must be in between 0 and 1 to set up correct neuron dropout rate.
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout should be a float between 0 and 1.")
    
    @staticmethod
    def __make_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Creates a convolutional block for feature extraction.
        
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :return: A sequential convolutional block containing convolution, batch normalization, ReLU activation, and max pooling.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),                                      # Extract features using 3×3 convolution
            nn.BatchNorm2d(out_channels),           # Normalize the output feature maps
            nn.ReLU(),                              # Introduce non-linearity
            nn.MaxPool2d(kernel_size=2, stride=2)   # Downsample feature maps by 2×2
        )
