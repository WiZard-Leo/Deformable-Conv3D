def load_checkpoint(checkpoint_path, model, optimizer, scaler, device='cuda:0'):
    """
    Load the checkpoint and restore the model, optimizer, and scaler states.
    
    Args:
    - checkpoint_path (str): Path to the checkpoint file.
    - model (torch.nn.Module): The model to load the state into.
    - optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    - scaler (torch.cuda.amp.GradScaler): The scaler to load the state into.
    - device (str): The device to map the loaded state to.
    
    Returns:
    - epoch (int): The epoch at which the checkpoint was saved.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore the optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore the scaler state
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Get the saved epoch
    epoch = checkpoint['epoch']
    
    return epoch

# Example usage:
if __name__ == "__main__":
    import sys
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    # sys.path.append('/home/ct/data/lwz/workplace/ct_detection/spatial-transformer-net/net')
    # sys.path.append('/home/ct/data/lwz/workplace/ct_detection/spatial-transformer-net/data')
    from net.deconv.DeformableNets import DeformVoxResNet

    # Define your model, optimizer, and scaler
    device = 'cpu'
    model = DeformVoxResNet(input_shape=(256,256,256), num_classes=5).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    scaler = torch.cuda.amp.GradScaler()

    # Load the checkpoint
    checkpoint_path = 'checkpoint/15.pth'
    epoch = load_checkpoint(checkpoint_path, model, optimizer, scaler, device=device)

    print(f'Checkpoint loaded, resuming training from epoch {epoch}')

    from data.dataset_npy import CTDataset, CTSubDataset
    train_dataset = CTDataset(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)
    for iter, (input_data, ct_label) in enumerate(train_loader):
        import pdb
        pdb.set_trace()
        # 获取类别索引
        class_idx = 1
        
        # 获取模型的卷积输出和分类输出
        conv_output, features, logits = model(input_data)

        # 获取类别激活权重
        class_activation_weights = model.fully_conn_3.weight.data[class_idx]

        # 对特征图进行全局平均池化
        global_avg_pool = F.adaptive_avg_pool3d(conv_output, (1, 1, 1))

        # 计算CAM
        cam = global_avg_pool * class_activation_weights.view(1, -1, 1, 1, 1)
        cam = torch.sum(cam, dim=1, keepdim=True)  # 保持通道维度

        # 将CAM上采样到输入尺寸
        cam = F.interpolate(cam, size=input_data.size()[2:], mode='trilinear', align_corners=True)

        # 将CAM转换为numpy数组
        cam_numpy = cam.cpu().detach().numpy()[0, 0, :, :, :]