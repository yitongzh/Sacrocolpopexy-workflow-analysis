# Load part of the pre-trained model
Endo3D_state_dict = model.state_dict()
pre_state_dict = torch.load('./params/params_endo3d_1vo1.pkl')
new_state_dict = {k: v for k, v in pre_state_dict.items() if k in Endo3D_state_dict}
Endo3D_state_dict.update(new_state_dict)
model.load_state_dict(Endo3D_state_dict)


# Split the networks for different learning rate
conv_params = list(map(id, model.conv1.parameters()))
conv_params += list(map(id, model.conv2.parameters()))
conv_params += list(map(id, model.conv3a.parameters()))
conv_params += list(map(id, model.conv3b.parameters()))
conv_params += list(map(id, model.conv4a.parameters()))
conv_params += list(map(id, model.conv4b.parameters()))
conv_params += list(map(id, model.conv5a.parameters()))
conv_params += list(map(id, model.conv5b.parameters()))
rest_params = filter(lambda x: id(x) not in conv_params, model.parameters())
conv_params = filter(lambda x: id(x) in conv_params, model.parameters())

optimizer = optim.Adam([{'params': conv_params, 'lr': 1e-7, 'weight_decay': 3e-5},
                        {'params': rest_params, 'lr': 1e-5, 'weight_decay': 3e-5}])


# Lock layers
for p in self.parameters():
    p.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6, weight_decay=3e-5)


# Visdom plotting
vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
                                           rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
                                           columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
vis.line(X=x, Y=np.column_stack((y_train_acc, y_batch_acc)), win='accuracy', update='append',
         opts=dict(title='accuracy', showlegend=True, legend=['train', 'valid']))
vis.line(X=x, Y=np.column_stack((y_train_loss, y_batch_loss)), win='loss', update='append',
         opts=dict(title='loss', showlegend=True, legend=['train', 'valid']))



