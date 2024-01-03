import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import timm
import os

from loader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from function import *
import wandb

class AugNet(nn.Module):
    def __init__(self, noise_lv):
        super(AugNet, self).__init__()
        ############# Trainable Parameters
        self.noise_lv = nn.Parameter(torch.zeros(1))
        self.shift_var = nn.Parameter(torch.empty(3,216,216))
        nn.init.normal_(self.shift_var, 1, 0.1)
        self.shift_mean = nn.Parameter(torch.zeros(3, 216, 216))
        nn.init.normal_(self.shift_mean, 0, 0.1)

        self.shift_var2 = nn.Parameter(torch.empty(3, 212, 212))
        nn.init.normal_(self.shift_var2, 1, 0.1)
        self.shift_mean2 = nn.Parameter(torch.zeros(3, 212, 212))
        nn.init.normal_(self.shift_mean2, 0, 0.1)

        self.shift_var3 = nn.Parameter(torch.empty(3, 208, 208))
        nn.init.normal_(self.shift_var3, 1, 0.1)
        self.shift_mean3 = nn.Parameter(torch.zeros(3, 208, 208))
        nn.init.normal_(self.shift_mean3, 0, 0.1)

        self.shift_var4 = nn.Parameter(torch.empty(3, 220, 220))
        nn.init.normal_(self.shift_var4, 1, 0.1)
        self.shift_mean4 = nn.Parameter(torch.zeros(3, 220, 220))
        nn.init.normal_(self.shift_mean4, 0, 0.1)

        # self.shift_var5 = nn.Parameter(torch.empty(3, 206, 206))
        # nn.init.normal_(self.shift_var5, 1, 0.1)
        # self.shift_mean5 = nn.Parameter(torch.zeros(3, 206, 206))
        # nn.init.normal_(self.shift_mean5, 0, 0.1)
        #
        # self.shift_var6 = nn.Parameter(torch.empty(3, 204, 204))
        # nn.init.normal_(self.shift_var6, 1, 0.5)
        # self.shift_mean6 = nn.Parameter(torch.zeros(3, 204, 204))
        # nn.init.normal_(self.shift_mean6, 0, 0.1)

        # self.shift_var7 = nn.Parameter(torch.empty(3, 202, 202))
        # nn.init.normal_(self.shift_var7, 1, 0.5)
        # self.shift_mean7 = nn.Parameter(torch.zeros(3, 202, 202))
        # nn.init.normal_(self.shift_mean7, 0, 0.1)

        self.norm = nn.InstanceNorm2d(3)

        ############## Fixed Parameters (For MI estimation
        self.spatial = nn.Conv2d(3, 3, 9).cuda()
        self.spatial_up = nn.ConvTranspose2d(3, 3, 9).cuda()

        self.spatial2 = nn.Conv2d(3, 3, 13).cuda()
        self.spatial_up2 = nn.ConvTranspose2d(3, 3, 13).cuda()

        self.spatial3 = nn.Conv2d(3, 3, 17).cuda()
        self.spatial_up3 = nn.ConvTranspose2d(3, 3, 17).cuda()


        self.spatial4 = nn.Conv2d(3, 3, 5).cuda()
        self.spatial_up4 = nn.ConvTranspose2d(3, 3, 5).cuda()


        # self.spatial5 = nn.Conv2d(3, 3, 19).cuda()
        # self.spatial_up5 = nn.ConvTranspose2d(3, 3, 19).cuda()
        # +
        # list(self.spatial5.parameters()) + list(self.spatial_up5.parameters())
        # #+
        #
        # self.spatial6 = nn.Conv2d(3, 3, 21).cuda()
        # self.spatial_up6 = nn.ConvTranspose2d(3, 3, 21).cuda()
        # list(self.spatial6.parameters()) + list(self.spatial_up6.parameters())
        # self.spatial7 = nn.Conv2d(3, 3, 23).cuda()
        # self.spatial_up7= nn.ConvTranspose2d(3, 3, 23).cuda()
        # list(self.spatial7.parameters()) + list(self.spatial_up7.parameters())
        self.color = nn.Conv2d(3, 3, 1).cuda()

        for param in list(list(self.color.parameters()) +
                          list(self.spatial.parameters()) + list(self.spatial_up.parameters()) +
                          list(self.spatial2.parameters()) + list(self.spatial_up2.parameters()) +
                          list(self.spatial3.parameters()) + list(self.spatial_up3.parameters()) +
                          list(self.spatial4.parameters()) + list(self.spatial_up4.parameters())
                          ):
            param.requires_grad=False

    def forward(self, x, estimation=False):
        if not estimation:
            spatial = nn.Conv2d(3, 3, 9).cuda()
            spatial_up = nn.ConvTranspose2d(3, 3, 9).cuda()

            spatial2 = nn.Conv2d(3, 3, 13).cuda()
            spatial_up2 = nn.ConvTranspose2d(3, 3, 13).cuda()

            spatial3 = nn.Conv2d(3, 3, 17).cuda()
            spatial_up3 = nn.ConvTranspose2d(3, 3, 17).cuda()

            spatial4 = nn.Conv2d(3, 3, 5).cuda()
            spatial_up4 = nn.ConvTranspose2d(3, 3, 5).cuda()

            # spatial5 = nn.Conv2d(3, 3, 19).cuda()
            # spatial_up5 = nn.ConvTranspose2d(3, 3, 19).cuda()
            #
            # spatial6 = nn.Conv2d(3, 3, 21).cuda()
            # spatial_up6 = nn.ConvTranspose2d(3, 3, 21).cuda()

            # spatial7 = nn.Conv2d(3, 3, 23).cuda()
            # spatial_up7 = nn.ConvTranspose2d(3, 3, 23).cuda()

            color = nn.Conv2d(3,3,1).cuda()
            weight = torch.randn(5)

            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(F.dropout(color(x), p=.2))

            x_sdown = spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(spatial_up(x_sdown))
            #
            x_s2down = spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(spatial_up2(x_s2down))
            #
            #
            x_s3down = spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(spatial_up3(x_s3down))

            #
            x_s4down = spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(spatial_up4(x_s4down))

            # x_s5down = spatial5(x)
            # x_s5down = self.shift_var5 * self.norm(x_s5down) + self.shift_mean5
            # x_s5 = torch.tanh(spatial_up5(x_s5down))+ weight[5] * x_s5

            # x_s6down = spatial6(x)
            # x_s6down = self.shift_var6 * self.norm(x_s6down) + self.shift_mean6
            # x_s6 = torch.tanh(spatial_up6(x_s6down))+ weight[6] * x_s6

            # x_s7down = spatial7(x)
            # x_s7down = self.shift_var7 * self.norm(x_s7down) + self.shift_mean7
            # x_s7 = torch.tanh(spatial_up7(x_s7down))+ weight[7] * x_s7

            output = (weight[0] * x_c + weight[1] * x_s + weight[2] * x_s2+ weight[3] * x_s3 + weight[4]*x_s4) / weight.sum()
        else:
            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(self.color(x))
            #
            x_sdown = self.spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(self.spatial_up(x_sdown))
            #
            x_s2down = self.spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(self.spatial_up2(x_s2down))

            x_s3down = self.spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(self.spatial_up3(x_s3down))

            x_s4down = self.spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(self.spatial_up4(x_s4down))

            # x_s5down = self.spatial5(x)
            # x_s5down = self.shift_var5 * self.norm(x_s5down) + self.shift_mean5
            # x_s5 = torch.tanh(self.spatial_up5(x_s5down)) + x_s5

            # x_s6down = self.spatial6(x)
            # x_s6down = self.shift_var6 * self.norm(x_s6down) + self.shift_mean6
            # x_s6 = torch.tanh(self.spatial_up6(x_s6down))+ x_s6

            # x_s7down = self.spatial7(x)
            # x_s7down = self.shift_var7 * self.norm(x_s7down) + self.shift_mean7
            # x_s7 = torch.tanh(self.spatial_up7(x_s7down))+ x_s7

            output = (x_c + x_s + x_s2 + x_s3 + x_s4) / 5
        return output
    

class EfficientNet_B0(nn.Module):

    def __init__(self, classes):
        super().__init__()
        self.feature_extractor = timm.create_model('efficientnet_b0', pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.p_logvar = nn.Sequential(nn.Linear(1280, 512),
                                      nn.ReLU())
        self.p_mu = nn.Sequential(nn.Linear(1280, 512),
                                  nn.LeakyReLU())
        self.class_classifier = nn.Linear(512, classes)

    def forward(self, x, train=True):
        x = self.feature_extractor.forward_features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        logvar = self.p_logvar(x)
        mu = self.p_mu(x)
        #
        end_points = {}
        end_points['logvar'] = logvar
        end_points['mu'] = mu

        if train:
            x = reparametrize(mu, logvar)
        else:
            x = mu
        end_points['Embedding'] = x
        x = self.class_classifier(x)
        end_points['Predictions'] = nn.functional.softmax(input=x, dim=-1)

        return x, end_points


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

class Trainer:
    def __init__(self, args, device, use_amp=True):
        self.args = args
        self.device = device
        self.counterk=0

        # Caffe Alexnet for singleDG task, Leave-one-out PACS DG task.
        # self.extractor = caffenet(args.n_classes).to(device)
        self.extractor = EfficientNet_B0(classes=args.n_classes).to(device)
        self.convertor = AugNet(1).cuda()

        self.source_loader = get_train_dataloader(args.IMG_TRAIN_DIR, args.train_batch_size)
        self.val_loader = get_val_dataloader(args.IMG_VAL_DIR, args.val_batch_size)
        self.target_loader = get_test_dataloader(args.IMG_TEST_DIR, args.val_batch_size)

        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)

        # Get optimizers and Schedulers, self.discriminator
        self.optimizer = torch.optim.SGD(self.extractor.parameters(), lr=self.args.learning_rate, nesterov=True, momentum=0.9, weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.args.epochs *0.8))

        self.convertor_opt = torch.optim.SGD(self.convertor.parameters(), lr=self.args.lr_sc)

        self.n_classes = args.n_classes
        self.centroids = 0
        self.d_representation = 0
        self.flag = False
        self.con = SupConLoss()
        self.target_id = None

        # Set up Automatic Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp


    def _do_epoch(self, epoch=None):
        criterion = nn.CrossEntropyLoss()
        self.extractor.train()
        tran = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for _, (data, class_l) in enumerate(self.source_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)

            loss_extractor, loss_convertor = 0, 0

            # Stage 1
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):

                self.optimizer.zero_grad()

                # Aug
                inputs_max = tran(torch.sigmoid(self.convertor(data)))
                inputs_max = inputs_max * 0.6 + data * 0.4
                data_aug = torch.cat([inputs_max, data])
                labels = torch.cat([class_l, class_l])

                # forward
                logits, tuple = self.extractor(data_aug)

                # Maximize MI between z and z_hat
                emb_src = F.normalize(tuple['Embedding'][:class_l.size(0)]).unsqueeze(1)
                emb_aug = F.normalize(tuple['Embedding'][class_l.size(0):]).unsqueeze(1)
                con = self.con(torch.cat([emb_src, emb_aug], dim=1), class_l)

                # Likelihood
                mu = tuple['mu'][class_l.size(0):]
                logvar = tuple['logvar'][class_l.size(0):]
                y_samples = tuple['Embedding'][:class_l.size(0)]
                likeli = -loglikeli(mu, logvar, y_samples)

                # Total loss & backward
                class_loss = criterion(logits, labels)
                loss_extractor = class_loss + self.args.alpha2*likeli + self.args.alpha1*con

            self.scaler.scale(loss_extractor).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # STAGE 2
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                inputs_max =tran(torch.sigmoid(self.convertor(data, estimation=True)))
                inputs_max = inputs_max * 0.6 + data * 0.4
                data_aug = torch.cat([inputs_max, data])

                # forward with the adapted parameters
                outputs, tuples = self.extractor(x=data_aug)

                # Upper bound MI
                mu = tuples['mu'][class_l.size(0):]
                logvar = tuples['logvar'][class_l.size(0):]
                y_samples = tuples['Embedding'][:class_l.size(0)]
                div = club(mu, logvar, y_samples)
                # div = criterion(outputs, labels)

                # Semantic consistency
                e = tuples['Embedding']
                e1 = e[:class_l.size(0)]
                e2 = e[class_l.size(0):]
                dist = conditional_mmd_rbf(e1, e2, class_l, num_class=self.args.n_classes)

                # Total loss and backward
                self.convertor_opt.zero_grad()
                loss_convertor = dist + self.args.beta * div

            self.scaler.scale(loss_convertor).backward()
            self.scaler.step(self.convertor_opt)
            self.scaler.update()

            print(f'[{epoch + 1}/{self.args.epochs}] Extractor loss: {loss_extractor} - Convertor Loss: {loss_convertor}')
            wandb.log({"Extractor Loss": loss_extractor, "Convertor Loss": loss_convertor})

        del loss_extractor, class_loss, logits

        self.extractor.eval()
        with torch.no_grad():
            avg_acc = self.do_test(self.val_loader)
            print(f'[{epoch + 1}/{self.args.epochs}] Evaluate Acc: {avg_acc}')
            wandb.log({"Epoch": epoch + 1, "Evaluate Acc": avg_acc})
            self.results["val"][self.current_epoch] = avg_acc

    def do_test(self, loader):
        class_correct = 0
        total = 0
        for _, (data, class_l) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)

            total += len(data)

            z = self.extractor(data, train=False)[0]


            _, cls_pred = z.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)

        return float(class_correct) / total

    def do_training(self):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        current_high = 0
        wandb.init(
            project='Domain Generalization', 
            name='EfficientNet_B0', 
            config={
                'epochs': self.args.epochs,

            }
        )
        for self.current_epoch in range(self.args.epochs):
            self._do_epoch(self.current_epoch)
            self.scheduler.step()
            if self.results["val"][self.current_epoch] > current_high:
                print('Saving Best model ...')
                torch.save(self.extractor.state_dict(), os.path.join(self.args.save_path, 'best_model.pth'))
                current_high = self.results["val"][self.current_epoch]
            if (self.current_epoch + 1) % 5 == 0:
                torch.save(self.extractor.state_dict(), os.path.join(self.args.save_path, f'epoch_{self.current_epoch + 1}.pth'))
                
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = test_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best test epoch: %g" % (
        val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        wandb.finish()

    def do_eval(self):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        current_high = 0
        self.logger.new_epoch(self.scheduler.get_lr())
        self.extractor.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)

                class_correct = self.do_test(loader)

                class_acc = float(class_correct) / total
                self.results[phase][0] = class_acc

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = test_res.argmax()
