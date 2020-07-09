import numpy as np
import tensorflow as tf
import scipy.stats
import util
import keras.backend as K
import tensorflow_probability as tfp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
from scipy.optimize import minimize

tfd = tfp.distributions
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

"""
This module contains the code for running the variety of optimizartion
strategies described in the paper
"""


def weighted_ml_opt(X_train, oracles, ground_truth, vae_0, weights_type='dbas',
                    LD=20, iters=20, samples=500, homoscedastic=False, homo_y_var=0.1,
                    quantile=0.95, verbose=False, alpha=1, train_gt_evals=None,
                    cutoff=1e-6, it_epochs=10, enc1_units=50):
    
    """
    Runs weighted maximum likelihood optimization algorithms ('CbAS', 'DbAS',
    RWR, and CEM-PI)
    """
    
    assert weights_type in ['cbas', 'dbas','rwr', 'cem-pi']
    L = X_train.shape[1]
    vae = util.build_vae(latent_dim=LD,
                    n_tokens=20, seq_length=L,
                    enc1_units=enc1_units)

    traj = np.zeros((iters, 7))
    oracle_samples = np.zeros((iters, samples))
    gt_samples = np.zeros((iters, samples))
    oracle_max_seq = None
    oracle_max = -np.inf
    gt_of_oracle_max = -np.inf
    y_star = -np.inf  
    
    for t in range(iters):
        ### Take Samples ###
        zt = np.random.randn(samples, LD)
        if t > 0:
            Xt_p = vae.decoder_.predict(zt)
            Xt = util.get_samples(Xt_p)
        else:
            Xt = X_train
        
        ### Evaluate ground truth and oracle ###
        yt, yt_var = util.get_balaji_predictions(oracles, Xt)
        if homoscedastic:
            yt_var = np.ones_like(yt) * homo_y_var
        Xt_aa = np.argmax(Xt, axis=-1)
        if t == 0 and train_gt_evals is not None:
            yt_gt = train_gt_evals
        else:
            yt_gt = ground_truth.predict(Xt_aa, print_every=1000000)[:, 0]
        
        ### Calculate weights for different schemes ###
        if t > 0:
            if weights_type == 'cbas': 
                log_pxt = np.sum(np.log(Xt_p) * Xt, axis=(1, 2))
                X0_p = vae_0.decoder_.predict(zt)
                log_px0 = np.sum(np.log(X0_p) * Xt, axis=(1, 2))
                w1 = np.exp(log_px0-log_pxt)
                y_star_1 = np.percentile(yt, quantile*100)
                if y_star_1 > y_star:
                    y_star = y_star_1
                w2= scipy.stats.norm.sf(y_star, loc=yt, scale=np.sqrt(yt_var))
                weights = w1*w2 
            elif weights_type == 'cem-pi':
                pi = scipy.stats.norm.sf(max_train_gt, loc=yt, scale=np.sqrt(yt_var))
                pi_thresh = np.percentile(pi, quantile*100)
                weights = (pi > pi_thresh).astype(int)
            elif weights_type == 'dbas':
                y_star_1 = np.percentile(yt, quantile*100)
                if y_star_1 > y_star:
                    y_star = y_star_1
                weights = scipy.stats.norm.sf(y_star, loc=yt, scale=np.sqrt(yt_var))
            elif weights_type == 'rwr':
                weights = np.exp(alpha*yt)
                weights /= np.sum(weights)
        else:
            weights = np.ones(yt.shape[0])
            max_train_gt = np.max(yt_gt)
            
        yt_max_idx = np.argmax(yt)
        yt_max = yt[yt_max_idx]
        if yt_max > oracle_max:
            oracle_max = yt_max
            try:
                oracle_max_seq = util.convert_idx_array_to_aas(Xt_aa[yt_max_idx-1:yt_max_idx])[0]
            except IndexError:
                print(Xt_aa[yt_max_idx-1:yt_max_idx])
            gt_of_oracle_max = yt_gt[yt_max_idx]
        
        ### Record and print results ##
        if t == 0:
            rand_idx = np.random.randint(0, len(yt), samples)
            oracle_samples[t, :] = yt[rand_idx]
            gt_samples[t, :] = yt_gt[rand_idx]
        if t > 0:
            oracle_samples[t, :] = yt
            gt_samples[t, :] = yt_gt
        
        traj[t, 0] = np.max(yt_gt)
        traj[t, 1] = np.mean(yt_gt)
        traj[t, 2] = np.std(yt_gt)
        traj[t, 3] = np.max(yt)
        traj[t, 4] = np.mean(yt)
        traj[t, 5] = np.std(yt)
        traj[t, 6] = np.mean(yt_var)
        
        if verbose:
            print(weights_type.upper(), t, traj[t, 0], color.BOLD + str(traj[t, 1]) + color.END, 
                  traj[t, 2], traj[t, 3], color.BOLD + str(traj[t, 4]) + color.END, traj[t, 5], traj[t, 6])
        
        ### Train model ###
        if t == 0:
            vae.encoder_.set_weights(vae_0.encoder_.get_weights())
            vae.decoder_.set_weights(vae_0.decoder_.get_weights())
            vae.vae_.set_weights(vae_0.vae_.get_weights())
        else:
            cutoff_idx = np.where(weights < cutoff)
            Xt = np.delete(Xt, cutoff_idx, axis=0)
            yt = np.delete(yt, cutoff_idx, axis=0)
            weights = np.delete(weights, cutoff_idx, axis=0)
            vae.fit([Xt], [Xt, np.zeros(Xt.shape[0])],
                  epochs=it_epochs,
                  batch_size=10,
                  shuffle=False,
                  sample_weight=[weights, weights],
                  verbose=0)
    
    max_dict = {'oracle_max' : oracle_max, 
                'oracle_max_seq': oracle_max_seq, 
                'gt_of_oracle_max': gt_of_oracle_max}
    return traj, oracle_samples, gt_samples, max_dict


def fb_opt(X_train, oracles, ground_truth, vae_0, weights_type='fbvae',
        LD=20, iters=20, samples=500, 
        quantile=0.8, verbose=False, train_gt_evals=None,
        it_epochs=10, enc1_units=50):
    
    """Runs FBVAE optimization algorithm"""
    
    assert weights_type in ['fbvae']
    L = X_train.shape[1]
    vae = util.build_vae(latent_dim=LD,
                    n_tokens=20, seq_length=L,
                    enc1_units=enc1_units)

    traj = np.zeros((iters, 7))
    oracle_samples = np.zeros((iters, samples))
    gt_samples = np.zeros((iters, samples))
    oracle_max_seq = None
    oracle_max = -np.inf
    gt_of_oracle_max = -np.inf
    y_star = - np.inf
    for t in range(iters):
        ### Take Samples and evaluate ground truth and oracle ##
        zt = np.random.randn(samples, LD)
        if t > 0:
            Xt_sample_p = vae.decoder_.predict(zt)
            Xt_sample = get_samples(Xt_sample_p)
            yt_sample, _ = get_balaji_predictions(oracles, Xt_sample)
            Xt_aa_sample = np.argmax(Xt_sample, axis=-1)
            yt_gt_sample = ground_truth.predict(Xt_aa_sample, print_every=1000000)[:, 0]
        else:
            Xt = X_train
            yt, _ = util.get_balaji_predictions(oracles, Xt)
            Xt_aa = np.argmax(Xt, axis=-1)
            fb_thresh = np.percentile(yt, quantile*100)
            if train_gt_evals is not None:
                yt_gt = train_gt_evals
            else:
                yt_gt = ground_truth.predict(Xt_aa, print_every=1000000)[:, 0]
        
        ### Calculate threshold ###
        if t > 0:
            threshold_idx = np.where(yt_sample >= fb_thresh)[0]
            n_top = len(threshold_idx)
            sample_arrs = [Xt_sample, yt_sample, yt_gt_sample, Xt_aa_sample]
            full_arrs = [Xt, yt, yt_gt, Xt_aa]
            
            for l in range(len(full_arrs)):
                sample_arr = sample_arrs[l]
                full_arr = full_arrs[l]
                sample_top = sample_arr[threshold_idx]
                full_arr = np.concatenate([sample_top, full_arr])
                full_arr = np.delete(full_arr, range(full_arr.shape[0]-n_top, full_arr.shape[0]), axis=0)
                full_arrs[l] = full_arr
            Xt, yt, yt_gt, Xt_aa = full_arrs
        yt_max_idx = np.argmax(yt)
        yt_max = yt[yt_max_idx]
        if yt_max > oracle_max:
            oracle_max = yt_max
            try:
                oracle_max_seq = util.convert_idx_array_to_aas(Xt_aa[yt_max_idx-1:yt_max_idx])[0]
            except IndexError:
                print(Xt_aa[yt_max_idx-1:yt_max_idx])
            gt_of_oracle_max = yt_gt[yt_max_idx]
        
        ### Record and print results ##

        rand_idx = np.random.randint(0, len(yt), samples)
        oracle_samples[t, :] = yt[rand_idx]
        gt_samples[t, :] = yt_gt[rand_idx]

        traj[t, 0] = np.max(yt_gt)
        traj[t, 1] = np.mean(yt_gt)
        traj[t, 2] = np.std(yt_gt)
        traj[t, 3] = np.max(yt)
        traj[t, 4] = np.mean(yt)
        traj[t, 5] = np.std(yt)
        if t > 0:
            traj[t, 6] = n_top
        else:
            traj[t, 6] = 0
        
        if verbose:
            print(weights_type.upper(), t, traj[t, 0], color.BOLD + str(traj[t, 1]) + color.END, 
                  traj[t, 2], traj[t, 3], color.BOLD + str(traj[t, 4]) + color.END, traj[t, 5], traj[t, 6])
        
        ### Train model ###
        if t == 0:
            vae.encoder_.set_weights(vae_0.encoder_.get_weights())
            vae.decoder_.set_weights(vae_0.decoder_.get_weights())
            vae.vae_.set_weights(vae_0.vae_.get_weights())
        else:
        
            vae.fit([Xt], [Xt, np.zeros(Xt.shape[0])],
                  epochs=1,
                  batch_size=10,
                  shuffle=False,
                  verbose=0)
            
    max_dict = {'oracle_max' : oracle_max, 
                'oracle_max_seq': oracle_max_seq, 
                'gt_of_oracle_max': gt_of_oracle_max}
    return traj, oracle_samples, gt_samples, max_dict



def killoran_opt(X_train, vae, oracles, ground_truth,
                 steps=10000, epsilon1=10**-5, epsilon2=1, noise_std=10**-5,
                 LD=100, verbose=False, adam=False):
    
    """Runs the Killoran optimization algorithm"""
    
    L = X_train.shape[1]
    
    G = vae.decoder_
    f = oracles
    
    sess = K.get_session()
    zt = K.tf.Variable(np.random.normal(size=[1, LD]), dtype='float32')
    pred_input = K.tf.Variable(np.zeros((1, L, X_train.shape[2])), dtype='float32')
    gen_output = G(zt)
    prior = tfd.Normal(0, 1)
    p_z = prior.log_prob(zt)
    predictions = K.tf.reduce_mean([f[i](pred_input)[0, 0] for i in range(len(f))])
    update_pred_input = K.tf.assign(pred_input, gen_output)
    dfdx = K.tf.gradients(ys=-predictions, xs=pred_input)[0]
    dfdz = K.tf.gradients(gen_output, zt, grad_ys=dfdx)[0]
    dpz = K.tf.gradients(p_z, zt)[0]
    
    noise = K.tf.random_normal(shape=[1, LD], stddev=noise_std)
    eps1 = K.tf.Variable(epsilon1, trainable=False)
    eps2 = K.tf.Variable(epsilon2, trainable=False)
    if adam:
        optimizer = K.tf.train.AdamOptimizer(learning_rate=epsilon2)
        step = dfdz + noise
    else:
        optimizer = K.tf.train.GradientDescentOptimizer(learning_rate=1)
        step = eps1 * dpz + eps2 * dfdz + noise
    
    design_op = optimizer.apply_gradients([(step, zt)])
    adam_initializers = [var.initializer for var in K.tf.global_variables() if 'Adam' in var.name or 'beta' in var.name]
    sess.run(adam_initializers)
    sess.run(pred_input.initializer)
    sess.run(zt.initializer)
    sess.run(eps1.initializer)
    sess.run(eps2.initializer)

    s = sess.run(K.tf.shape(zt))
    sess.run(update_pred_input, {zt: np.random.normal(size=s)})
    z_0 = sess.run([zt])
    results = np.zeros((steps, 2))
    xt_prev = None
    for t in range(steps):
        xt0, _, = sess.run([gen_output, design_op], {eps1: epsilon1, eps2:epsilon2})
        pred_in, preds = sess.run([update_pred_input, predictions])
        xt = util.get_argmax(xt0)
        ft = util.get_balaji_predictions(oracles, xt)[0][0]
        xt_seq = np.argmax(xt, axis=-1)
        if xt_prev is None or not np.all(xt_seq == xt_prev):
            xt_prev = xt_seq
            gt = ground_truth.predict(xt_seq)[:, 0][0]
        else:
            gt = results[t-1, 1]
        results[t, 0] = ft
        results[t, 1] = gt
    return results, {}


def gomez_bombarelli_opt(X_train, pred_vae, ground_truth, total_it=10000, constrained = True):
    """Runs the Gomez-Bombarelli optimization algorithm"""
    LD = 20
    L = X_train.shape[1]
    
    embeddings = pred_vae.encoder_.predict([X_train])[0]
    predictions = pred_vae.predictor_.predict(embeddings)[0]
    predictions = predictions.reshape((predictions.shape[0]))
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)
    print("Fitting GP...")
    gp.fit(embeddings,predictions)
    print("Finishing fitting GP")
    
    def gp_z_estimator(z):
        val = gp.predict(z.reshape(1,-1)).ravel()[0]
        return -val 

    method='COBYLA'
    num_it = 0
    f_opts = []
    gt_opts = []
    
    z0 = np.random.randn(LD)
    if constrained:
        z_norm = np.linalg.norm(embeddings, axis=1)
        z_norm_mean = np.mean(z_norm)
        z_norm_std = np.std(z_norm)

        lower_rad = z_norm_mean - z_norm_std*2
        higher_rad = z_norm_mean + z_norm_std*2
        constraints = [{'type': 'ineq', 'fun': lambda x:  np.linalg.norm(x)-lower_rad },
                   {'type': 'ineq', 'fun': lambda x:  higher_rad-np.linalg.norm(x) }]
        res = minimize(gp_z_estimator, z0, method=method, constraints=constraints,
                       options={'disp': False,'tol':0.1,'maxiter':10000},tol=0.1)
    else:
        res = minimize(gp_z_estimator, z0, method=method,
                       options={'disp': False,'tol':0.1,'maxiter':10000},tol=0.1)
    z_opt = res.x.reshape((1, LD))
    f_opt = -res.fun
    
    x_opt = np.argmax(pred_vae.decoder_.predict(z_opt), axis=-1)
    gt_opt = ground_truth.predict(x_opt)[:, 0][0]
    print(f_opt, gt_opt)
    
    oracle_max_seq = util.convert_idx_array_to_aas(x_opt)
    
    max_dict = {'oracle_max' : f_opt, 
                'oracle_max_seq': oracle_max_seq, 
                'gt_of_oracle_max': gt_opt}

    results = np.array([f_opt, gt_opt])
    return results, max_dict