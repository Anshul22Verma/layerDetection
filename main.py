import argparse
import os
import pandas as pd

# Argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="BYOL Pretraining")
    
    # pre-training arguments
    parser.add_argument('--ssl_method', type=str, required=False, choices=['MoCo', 'SimSiam', 'SimCLR', 'BYOL', None], default=None, 
                        help="Picks a pipeline to use for training")
    
    parser.add_argument('--ssl_image_root', type=str, required=False, help="Path to the root directory with all the unlabelled images")
    parser.add_argument('--ssl_epochs', type=int, required=False, help="Epochs for self-supervised learning")
    parser.add_argument('--ssl_projection_dim', type=int, default=128, help="Projection head output dimension")
    parser.add_argument('--ssl_hidden_dim', type=int, default=512, help="Hidden layer dimension in the projection and predictor heads")
    parser.add_argument('--ssl_lr', type=float, default=0.001, help="Learning rate for self-supervised learning")
    parser.add_argument('--ssl_batch_size', type=int, default=64, help="Batch size for fine-tuning")
    
    # common arguments
    parser.add_argument('--result_dir', type=str, required=True, help="Directory to save all the models in")
    parser.add_argument('--architecture', type=str, required=True, help="Architecture picked as base model")
    
    # fine-tuning arguments
    parser.add_argument('--image_root', type=str, required=True, help="Directory to to load fine-tuning images from")
    parser.add_argument('--image_labels', type=str, required=True, help="File to to load fine-tuning labels from a csv file")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs in fine-tuning")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for fine-tuning")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for fine-tuning")
    return parser.parse_args()


# Main training function
def main():
    # Parse arguments
    args = parse_args()
    
    ft_images = []
    ft_labels = []
    uq_labels = set()
    image_root = args.image_root
    df = pd.read_csv(args.image_labels)
    for row in df:
        ft_images.append(os.path.join(image_root, row["image"]))
        ft_labels.append(row["label"].split(","))
        for l in row["label"].split(","):
            if len(l) > 0:
                uq_labels.add(l)
    num_classes = len(uq_labels)
    print(f"Number of classes in the dataset are {num_classes}, {uq_labels}")
    uq_labels = list(uq_labels)
    # create the result-directory if it doesn't exist
    os.makedirs(args.result_dir, exist_ok=True)

    if args.ssl_method:
        # perform pre-training
        images = [img for img in os.listdir(args.ssl_image_root) if ".png" in img]
        match args.ssl_method:
            case "MoCo":
                # MoCo pipeline
                from trainer.moco_trainer import pre_train_model, train_classifier_w_pretraining
                model = pre_train_model(architecture=args.architecture, hidden_dim=args.ssl_hidden_dim, 
                                        projection_dim=args.ssl_projection_dim, queue_size=1024*6, momentum=0.99,
                                        pretrained=True, images=images, batch_size=args.ssl_batch_size, num_workers=2,
                                        model_dir=args.result_dir, lr=args.ssl_lr, temperature=0.5, memory_size=1024*2, 
                                        epochs=args.ssl_epochs, model_name=f"MoCo_{args.architecture}.pth")
                model = train_classifier_w_pretraining(pre_trained_model=model, num_classes=num_classes,
                                                       image_paths=ft_images, labels=ft_labels, uq_classes=uq_labels,
                                                       lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, 
                                                       model_dir=args.result_dir)
            case "SimSiam":
                # SimSiam pipeline
                from trainer.sim_siam_trainer import pre_train_model, train_classifier_w_pretraining
                model = pre_train_model(architecture=args.architecture, hidden_dim=args.ssl_hidden_dim, 
                                        projection_dim=args.ssl_projection_dim, queue_size=1024*6, momentum=0.99,
                                        pretrained=True, images=images, batch_size=args.ssl_batch_size, num_workers=2,
                                        model_dir=args.result_dir, lr=args.ssl_lr, 
                                        epochs=args.ssl_epochs, model_name=f"SimSiam_{args.architecture}.pth")
                model = train_classifier_w_pretraining(pre_trained_model=model, num_classes=num_classes,
                                                       image_paths=ft_images, labels=ft_labels, uq_classes=uq_labels,
                                                       lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, 
                                                       model_dir=args.result_dir)
            case "SimCLR":
                # SimCLR pipeline
                from trainer.contrastive_trainer import pre_train_model, train_classifier_w_pretraining
                model = pre_train_model(architecture=args.architecture, embedding_dim=args.ssl_hidden_dim, pretrained=True, 
                                        images=images, batch_size=args.ssl_batch_size, num_workers=2,
                                        model_dir=args.result_dir, lr=args.ssl_lr, temperature=0.5, 
                                        epochs=args.ssl_epochs, model_name=f"SimCLR_{args.architecture}.pth")
                model = train_classifier_w_pretraining(pre_trained_model=model, num_classes=num_classes,
                                                       image_paths=ft_images, labels=ft_labels, uq_classes=uq_labels,
                                                       lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, 
                                                       model_dir=args.result_dir)
            case "BYOL":
                # BYOL pipeline
                from trainer.byol_trainer import pre_train_model, train_classifier_w_pretraining
                model = pre_train_model(architecture=args.architecture, hidden_dim=args.ssl_hidden_dim, 
                                        projection_dim=args.ssl_projection_dim, queue_size=1024*6, momentum=0.99, pretrained=True, 
                                        images=images, batch_size=args.ssl_batch_size, num_workers=2,
                                        model_dir=args.result_dir, lr=args.ssl_lr, 
                                        epochs=args.ssl_epochs, model_name=f"BYOL_{args.architecture}.pth")
                model = train_classifier_w_pretraining(pre_trained_model=model, num_classes=num_classes,
                                                       image_paths=ft_images, labels=ft_labels, uq_classes=uq_labels,
                                                       lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, 
                                                       model_dir=args.result_dir)
    else:
        # No pre-training just fine-tuning pipeline
        from trainer.classification import train_classifier
        model = train_classifier(architecture=args.architecture, num_classes=num_classes, image_paths=ft_images,
                                labels=ft_labels, uq_classes=uq_labels, lr=args.lr, batch_size=args.batch_size, 
                                epochs=args.epochs, model_dir=args.result_dir)

    # -------- TESTING -------- #