from torchvision import transforms


def basic_classification_augmentation():
    return transforms.Compose([
            transforms.Resize((756, 756)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # Strong blur
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class MoCoAugment:
    """ MoCo v2: Strong augmentations for query, weak for key. """
    def __init__(self):
        self.strong = transforms.Compose([
            transforms.Resize((756, 756)),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),  # Strong random crop
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.weak = transforms.Compose([
            transforms.Resize((756, 756)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        return self.strong(img), self.weak(img)


class SimCLRAugment:
    """ Strong augmentations as per the original SimCLR paper. """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((756, 756)),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),  # Large scale variation
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),  # Strong blur
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return self.transform(img), self.transform(img)


class SimSiamAugment:
    """ SimSiam augmentations as per the paper. """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((756, 756)),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),  # Larger scale range
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return self.transform(img), self.transform(img)


class BYOLAugment:
    """ BYOL augmentations as per the paper. """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((756, 756)),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),  # Strong augmentation
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        return self.transform(img), self.transform(img)