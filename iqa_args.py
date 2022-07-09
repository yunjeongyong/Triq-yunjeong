args = {
    'name': 'test0',
    'model_type': 'R50-ViT-B_16',
    'csv_path': 'C:\\Users\\yunjeongyong\\Desktop\\intern\\Triq-yunjeong\\data\\all_data_csv\\KonIQ-10k.txt.csv',
    'data_path': 'C:\\Users\\yunjeongyong\\Desktop\\intern\\Triq-yunjeong\\data\\1024x768',
    'pretrained_dir': r'.\pretrained_weights\imagenet21k+imagenet2012_R50+ViT-B_16.npz',
    'output_dir': r'C:\Users\yunjeongyong\Desktop\intern\Triq-yunjeong\data\results',
    'snapshot_dir': 'C:\\Users\\yunjeongyong\\Desktop\\intern\\Triq-yunjeong\\data\\snapshot',
    'train_batch_size': 8,
    'test_batch_size': 8,
    'eval_every': 4462,
    'learning_rate': 1e-4 / 2,
    'weight_decay': 0,
    'epoch': 120,
    'decay_type': 'cosine',
    'warmup_steps': 10,
    'max_grad_norm': 1.,
    'local_rank': 0,
    'seed': 32,
    'gradient_accumulation_steps': 1,
    'fp16': False,
    'loss_scale': 0,
    'batch_size': 8
}

# args['model_type'] = 'ViT-B_16'
# args['pretrained_dir'] = r'.\pretrained_weights\imagenet21k+imagenet2012_ViT-B_16.npz'
