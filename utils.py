from torchvision import datasets, transforms

def get_transforms(image_size):
    transform = transforms.Compose([        
            transforms.ToTensor(),
            transforms.Resize((image_size,image_size),antialias=False),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    invTrans = transforms.Compose([
                            transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                    transforms.ToPILImage()
                                ])
    return transform,invTrans