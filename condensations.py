import torch
import copy

from torch.nn import CrossEntropyLoss
from torch.nn import CrossEntropyLoss

from avalanche.benchmarks.utils import AvalancheTensorDataset
from utils import save_images, log_metrics, get_statistics


def distance_wb(self, gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def match_loss(self, gw_syn, gw_real, device):
    dis = torch.tensor(0.0).to(device)

    for ig in range(len(gw_real)):
        gwr = gw_real[ig]
        gws = gw_syn[ig]
        dis += distance_wb(self, gwr, gws)
    return dis


def update_network(self, image_syn, label_syn, strategy, optimizer, model):
    exp_counter = strategy.training_exp_counter
    image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())

    dst_syn_train = AvalancheTensorDataset(image_syn_train,
                                        label_syn_train)
    trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=len(image_syn_train), shuffle=True, num_workers=0)
    criterion = CrossEntropyLoss().to(strategy.device)
    model = model.to(strategy.device)
    for _ in range(self.inner_loop):
        for (mb_x, mb_y, _) in trainloader:
            mb_x.to(strategy.device)
            mb_y.to(strategy.device)
            output = model(mb_x)
            loss = criterion(output, mb_y)
            loss_name = 'Loss_MB/InnerLoop/Task' + str(exp_counter)
            log_metrics(self.wandb_logger, loss.detach().item(), loss_name)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def condenseImagesLinearComb(self, real_imgs, num_condensed_imgs, strategy, log=''):

    def getImagesLinearComb(weights, images, mask, device):
        images = images.to(device)
        final_shape = list(images.shape)
        res = weights * mask
        res = res.mm(images.view(final_shape[0], -1))
        final_shape[0] = weights.shape[0]
        return res.view(final_shape)

    def get_all_condensed_images(weights_dict, images, classes, masks, device):
        imgs = torch.tensor([],device=device)
        for c in classes:
            img = getImagesLinearComb(weights_dict[c], images[c][:][0], masks[c],device)
            imgs = torch.cat((imgs,img), dim=0)
        return imgs

    def normalize_coefficients(weights, mask):
        for i in range(len(weights)):
            with torch.no_grad():
                if torch.sum(weights[i]) > 0:
                    # w_mask = weights[i] * mask[i]
                    weights[i] = weights[i] / torch.sum(weights[i])
        return weights

    def mini_batch_update(self, strategy, real_imgs, net, criterion,
                        weights_dict, loss, c, net_parameters, masks):
        exp_counter = strategy.training_exp_counter
        img_real, lab_real, _ = real_imgs[c][:]
        img_real = img_real.to(strategy.device)
        lab_real = lab_real.to(strategy.device)
        output_real = net(img_real)
        loss_real = criterion(output_real, lab_real)
        loss_name = 'Loss_MB/OuterLoop/NetReal/Task' + \
            str(exp_counter)
        log_metrics(self.wandb_logger, loss_real.detach().item(), loss_name)
        gw_real = torch.autograd.grad(loss_real, net_parameters)
        gw_real = list((_.detach().clone() for _ in gw_real))

        img_syn = getImagesLinearComb(weights_dict[c], real_imgs[c][:][0], masks[c], strategy.device)
        lab_syn = torch.ones((len(img_syn),), device=strategy.device,
                                dtype=torch.long) * c
        output_syn = net(img_syn)
        loss_syn = criterion(output_syn, lab_syn)

        loss_name = 'Loss_MB/OuterLoop/NetSyn/Task' + \
            str(exp_counter)
        log_metrics(self.wandb_logger, loss_syn.detach().item(), loss_name)

        gw_syn = torch.autograd.grad(loss_syn, net_parameters,
                                    create_graph=True,
                                    retain_graph=True)
        loss += match_loss(self, gw_syn, gw_real, strategy.device)

    def outer_loop_body(self, strategy, net, real_imgs, weights_dict, criterion,
                        classes, optimizer_weights, optimizer_net, net_parameters, masks):

        label_syn = torch.cat([torch.ones(num_condensed_imgs[c], dtype = torch.long) * c for c in classes]).view(-1)
        label_syn.requires_grad = False
        label_syn = label_syn.to(strategy.device)
        exp_counter = strategy.training_exp_counter

        for ol in range(self.outer_loop):
            ''' update synthetic data '''

            loss = torch.tensor(0.0).to(strategy.device)
            for c in classes:
                mini_batch_update(self, strategy, real_imgs, net, criterion,
                                    weights_dict, loss, c, net_parameters, masks)

            loss_name = 'Loss_MB/OuterLoop/Images/Task' + \
                str(exp_counter)
            log_metrics(self.wandb_logger, loss.detach().item() / len(classes),
                        loss_name)

            optimizer_weights.zero_grad()
            loss.backward()
            optimizer_weights.step()

            with torch.no_grad():
                for c in weights_dict.keys():
                    neg_ids = weights_dict[c] <= 0
                    weights_dict[c][neg_ids] = 0
                    rows_zeros = torch.all(neg_ids, dim=1)
                    if torch.any(rows_zeros):
                        idxs_row = torch.nonzero(rows_zeros).squeeze(dim=1).tolist()
                        for id_row in idxs_row:
                            idxs_ones = torch.nonzero(masks[c][id_row]).squeeze()
                            i = torch.randint(low=0, high=len(idxs_ones) - 1, size=(1,) )
                            weights_dict[c][id_row][idxs_ones[i]] = 1.0


            if ol == self.outer_loop - 1:
                break

            with torch.no_grad():
                weights_dict_cp = copy.deepcopy(weights_dict)
                for key in weights_dict_cp.keys():
                    weights_dict_cp[key] = normalize_coefficients(weights_dict_cp[key], masks[key])
                imgs_syn = get_all_condensed_images(weights_dict_cp, real_imgs, classes,masks, strategy.device)
            net = update_network(self, imgs_syn, label_syn, strategy,
                                    optimizer_net, net)

    def iterationBody(self, strategy, real_imgs, weights_dict, criterion,
                    classes, optimizer_weights, masks):
        for _ in range(self.iteration):
            net = copy.deepcopy(strategy.model)
            net = net.to(strategy.device)
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=self.lr_net)  #  optim for the net
            optimizer_net.zero_grad()
            
            outer_loop_body(
                self, strategy, net, real_imgs, weights_dict, criterion,
                classes, optimizer_weights, optimizer_net, net_parameters, masks)


    criterion = CrossEntropyLoss()
    classes = sorted(list(real_imgs.keys()))
    exp_counter = strategy.training_exp_counter
    weights_dict = {}
    masks = {}
    for c in classes:
        #  Random initialization of the coefficient used for the linear combination
        weight = torch.rand(
            size=(num_condensed_imgs[c], len(real_imgs[c])), dtype=torch.float,
            requires_grad=True, device=strategy.device)
        weights_dict[c] = weight

        '''
        Each condensed image is composed by a different set of starting
        images. The mask as a filter to build the disjoint initial sets of
        each condensed image.

        Mask example:

        [1 1 0 0 0 0 0 0 0 0]
        [0 0 1 1 0 0 0 0 0 0]
        [0 0 0 0 1 1 0 0 0 0]
        [0 0 0 0 0 0 1 1 0 0]
        [0 0 0 0 0 0 0 0 1 1]
        '''

        mask = torch.zeros(size=weight.shape,requires_grad=False,device=strategy.device)
        for i in range(len(mask)):
            indexes = [2 * i, 2 * i + 1]
            mask[i][indexes] = 1.0
        masks[c] = mask

    with torch.no_grad():
        for c in classes:
            weights_dict[c] = weights_dict[c] * masks[c]
            weights_dict[c] = normalize_coefficients(weights_dict[c], masks[c])
            weights_dict[c].requires_grad = True

    if self.debug:
        with torch.no_grad():
            imgs = get_all_condensed_images(weights_dict, real_imgs, classes, masks,strategy.device)
            save_images(self.dataset, self.mem_size, imgs,
                        log + '_' + str(exp_counter) + '_start', 'single', self.wandb_logger, exp_counter)

    optimizer_weights = torch.optim.SGD(
        [weights for weights in weights_dict.values()],
        lr=self.lr_w, weight_decay=self.l2_w, momentum=0.5
        )
    optimizer_weights.zero_grad()
    
    iterationBody(self, strategy, real_imgs, weights_dict, criterion,
                    classes, optimizer_weights, masks)

    condensed_datasets = {}
    for c in classes:
        weights_dict[c] = normalize_coefficients(weights_dict[c], masks[c])
        imgs = getImagesLinearComb(weights_dict[c], real_imgs[c][:][0], masks[c], strategy.device).detach()
        imgs = imgs.to('cpu')
        condensed_datasets[c] = AvalancheTensorDataset(imgs, [c for _ in range(len(imgs))])

    if self.debug:
        with torch.no_grad():
            imgs = get_all_condensed_images(weights_dict, real_imgs, classes, masks, strategy.device)
            save_images(self.dataset, self.mem_size, imgs,
                        log + '_' + str(exp_counter) + '_end', 'single', self.wandb_logger, exp_counter)
            get_statistics(weights_dict, masks, self.statistics, exp_counter)

    return condensed_datasets


def condenseImagesOriginalGradientMatching(self, real_imgs, num_condensed_imgs, strategy, log=''):
    criterion = CrossEntropyLoss()  # .to(strategy.device)
    classes = sorted(list(real_imgs.keys()))
    num_classes = len(classes)
    total_imgs = sum(num_condensed_imgs.values())
    image_syn = torch.randn(size=(total_imgs, self.image_size[0],
                                self.image_size[1], self.image_size[2]),
                            dtype=torch.float, requires_grad=True)
    label_syn = torch.tensor([], dtype=torch.long,
                            requires_grad=False,
                            device=strategy.device)
    for c in classes:
        label_syn = torch.cat((label_syn, torch.ones(size=(num_condensed_imgs[c],), dtype=torch.long,device=strategy.device) * c))

    for index, c in enumerate(classes):
        idx_shuffle = torch.randperm(len(real_imgs[c]))[:num_condensed_imgs[c]]
        ids = label_syn == c
        image_syn.data[ids] = real_imgs[c][idx_shuffle][0].detach().data

    optimizer_img = torch.optim.SGD([image_syn, ], lr=self.lr_w,
                                    momentum=0.5)
    optimizer_img.zero_grad()
    image_syn = image_syn.to(strategy.device)
    save_images(self.dataset, self.mem_size, image_syn,
                log + '_' + str(strategy.training_exp_counter) + 'start', 'single', self.wandb_logger,
                strategy.training_exp_counter)
    for it in range(self.iteration):

        ''' Train synthetic data '''

        net = copy.deepcopy(strategy.model)
        net.train()
        net = net.to(strategy.device)
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=self.lr_net,
                                        momentum=0.5) 
        optimizer_net.zero_grad()
        loss_avg = 0

        for ol in range(self.outer_loop):
            
            # Code to get stabl mu and sigma values,
            # making the training of the Batch normalization easier
            BN_flag = False
            BNSizePC = 16 
            for module in net.modules():
                if 'BatchNorm' in module._get_name(): #BatchNorm
                    BN_flag = True
            if BN_flag:
                img_real = []
                for c in classes:
                    ids = torch.randperm(len(real_imgs[c]))[:BNSizePC]
                    img_real.append(real_imgs[c][ids][0])
                img_real = torch.cat(img_real, dim=0)
                net.train() # for updating the mu, sigma of BatchNorm
                img_real = img_real.to(strategy.device)
                output_real = net(img_real) # get running mu, sigma
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  #BatchNorm
                        module.eval() # fix mu and sigma of every BatchNorm layer

            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(strategy.device)
            for index, c in enumerate(classes):
                img_real, lab_real, _ = real_imgs[c][:]
                img_real = img_real.to(strategy.device)
                lab_real = lab_real.to(strategy.device)
                output_real = net(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                ids = label_syn == c
                img_syn = image_syn[ids].reshape((num_condensed_imgs[c], self.image_size[0], self.image_size[1], self.image_size[2]))
                lab_syn = label_syn[ids]
                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                loss += match_loss(self,gw_syn, gw_real, strategy.device)
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()

            if ol == self.outer_loop - 1:
                break

            update_network(self, image_syn, label_syn, strategy, optimizer_net, net)

        loss_avg /= (num_classes*self.outer_loop)


    save_images(self.dataset, self.mem_size, image_syn,
                log + '_' + str(strategy.training_exp_counter) + 'end', 'single', self.wandb_logger,
                strategy.training_exp_counter)
    condensed_datasets = {}

    for i, c in enumerate(classes):
        ids = label_syn == c
        condensed_datasets[c] = AvalancheTensorDataset(
            image_syn[ids].detach().to('cpu'), label_syn[ids].tolist())
    return condensed_datasets
