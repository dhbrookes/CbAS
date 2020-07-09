import warnings
warnings.filterwarnings("ignore")
import numpy as np
import keras.callbacks
import keras.utils.generic_utils
import keras.models
import json
import keras.backend as K
import tensorflow as tf

import util
import gfp_gp
import losses
import optimization_algs 


"""
This module contains the code to run the tests of optimization
methods on the GFP brightness data
"""

def train_and_save_oracles(X_train, y_train, n=10, suffix='', batch_size=100):
    """Trains a set of n oracles on a given set of data"""
    for i in range(n):
        model = util.build_pred_model(n_tokens=20, seq_length=X_train.shape[1], enc1_units=50)
        model.compile(optimizer='adam',
                      loss=losses.neg_log_likelihood,
                      )
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   min_delta=0, 
                                                   patience=5, 
                                                   verbose=1)

        model.fit(X_train, y_train, 
                  epochs=100, 
                  batch_size=batch_size, 
                  validation_split=0.1, 
                  callbacks=[early_stop],
                  verbose=2)
        model.save("../models/oracle_%i%s.h5" % (i, suffix))
        
        
def evaluate_ground_truth(X_aa, ground_truth, save_file=None):
    """Evaluates the GFP ground truth function, i.e. the GP model"""
    y_gt = ground_truth.predict(X_aa, print_every=100000)[:, 0]
    if save_file is not None:
        np.save(save_file, y_gt)
        
        
def train_experimental_oracles():
    """
    Trains and saves oracles on the simulated GFP data (i.e. data generated
    from the GP model
    """
    TRAIN_SIZE = 5000
    train_size_str = "%ik" % (TRAIN_SIZE/1000)
    i = 1
    num_models = [1, 5, 20]
    for i in range(len(num_models)):
        RANDOM_STATE = i+1
        nm = num_models[i]
        X_train, y_train, _  = util.get_experimental_X_y(random_state=RANDOM_STATE, train_size=TRAIN_SIZE)
        suffix = '_%s_%i_%i' % (train_size_str, nm, RANDOM_STATE)
        train_and_save_oracles(X_train, y_train, batch_size=10, n=nm, suffix=suffix)
        
        
def train_experimental_vaes():
    """Trains and saves VAEs on the GFP data for use in the weighted ML methods"""
    TRAIN_SIZE = 5000
    train_size_str = "%ik" % (TRAIN_SIZE/1000)
    suffix = '_%s' % train_size_str
    for i in [0, 2]:
        RANDOM_STATE = i + 1
        X_train, _, _  = util.get_experimental_X_y(random_state=RANDOM_STATE, train_size=TRAIN_SIZE)
        vae_0 = util.build_vae(latent_dim=20,
                  n_tokens=20, 
                  seq_length=X_train.shape[1],
                  enc1_units=50)
        vae_0.fit([X_train], [X_train, np.zeros(X_train.shape[0])],
                  epochs=100,
                  batch_size=10,
                  verbose=2)
        vae_0.encoder_.save_weights("../models/vae_0_encoder_weights%s_%i.h5"% (suffix, RANDOM_STATE))
        vae_0.decoder_.save_weights("../models/vae_0_decoder_weights%s_%i.h5"% (suffix, RANDOM_STATE))
        vae_0.vae_.save_weights("../models/vae_0_vae_weights%s_%i.h5"% (suffix, RANDOM_STATE))
        

def train_experimental_pred_vaes():
    """
    Trains and saves the semi-supervised VAEs on the GFP experimental data for use 
    in the Gomez-bombarelli optimization method.
    """
    TRAIN_SIZE = 5000
    train_size_str = "%ik" % (TRAIN_SIZE/1000)
    for it in range(3):
        RANDOM_STATE = it + 1
        X_train, y_train, _  = util.get_experimental_X_y(random_state=RANDOM_STATE, train_size=TRAIN_SIZE)
        
        L = X_train.shape[1]
        LD=20
        gt_var=0.01
        pred_vae = util.build_pred_vae_model(latent_dim=LD, n_tokens=X_train.shape[2], 
                                            seq_length=L, enc1_units=50, pred_var=gt_var)

        pred_vae.fit([X_train], [X_train, np.zeros(X_train.shape[0]), y_train, np.zeros_like(y_train)],
                      batch_size=10,
                      epochs=100,
                      shuffle=True,
                      validation_split=0,
                      verbose=2
                    )
        suffix = "_%s_%i" % (train_size_str, RANDOM_STATE)
        pred_vae.encoder_.save_weights("../models/pred_vae_encoder_weights%s.h5" % suffix)
        pred_vae.decoder_.save_weights("../models/pred_vae_decoder_weights%s.h5" % suffix)
        pred_vae.predictor_.save_weights("../models/pred_vae_predictor_weights%s.h5" % suffix)
        pred_vae.vae_.save_weights("../models/pred_vae_vae_weights%s.h5" % suffix)
        
        
def run_experimental_weighted_ml(it, repeats=3):
    """Runs the GFP comparative tests on the weighted ML models and FBVAE."""
    
    assert it in [0, 1, 2]
    
    TRAIN_SIZE = 5000
    train_size_str = "%ik" % (TRAIN_SIZE/1000)
    num_models = [1, 5, 20][it]
    RANDOM_STATE = it + 1
    
    X_train, y_train, gt_train  = util.get_experimental_X_y(random_state=RANDOM_STATE, train_size=TRAIN_SIZE)
    
    vae_suffix = '_%s_%i' % (train_size_str, RANDOM_STATE)
    oracle_suffix = '_%s_%i_%i' % (train_size_str, num_models, RANDOM_STATE)
    
    vae_0 = build_vae(latent_dim=20,
                  n_tokens=20, 
                  seq_length=X_train.shape[1],
                  enc1_units=50)

    vae_0.encoder_.load_weights("../models/vae_0_encoder_weights%s.h5" % vae_suffix)
    vae_0.decoder_.load_weights("../models/vae_0_decoder_weights%s.h5"% vae_suffix)
    vae_0.vae_.load_weights("../models/vae_0_vae_weights%s.h5"% vae_suffix)
    
    ground_truth = gfp_gp.SequenceGP(load=True, load_prefix="data/gfp_gp")
    
    loss = neg_log_likelihood
    keras.utils.generic_utils.get_custom_objects().update({"neg_log_likelihood": loss})
    oracles = [keras.models.load_model("../models/oracle_%i%s.h5" % (i, oracle_suffix)) for i in range(num_models)]
    
    test_kwargs = [
                   {'weights_type':'cbas', 'quantile': 1},
                   {'weights_type':'rwr', 'alpha': 20},
                   {'weights_type':'dbas', 'quantile': 0.95},
                   {'weights_type':'cem-pi', 'quantile': 0.8},
                   {'weights_type': 'fbvae', 'quantile': 0.8}
    ]
    
    base_kwargs = {
        'homoscedastic': False,
        'homo_y_var': 0.01,
        'train_gt_evals':gt_train,
        'samples':100,
        'cutoff':1e-6,
        'it_epochs':10,
        'verbose':True,
        'LD': 20,
        'enc1_units':50,
        'iters': 50
    }
    
    if num_models==1:
        base_kwargs['homoscedastic'] = True
        base_kwargs['homo_y_var'] = np.mean((util.get_balaji_predictions(oracles, X_train)[0] - y_train)**2)
    
    for k in range(repeats):
        for j in range(len(test_kwargs)):
            test_name = test_kwargs[j]['weights_type']
            suffix = "_%s_%i_%i" % (train_size_str, RANDOM_STATE, k)
            if test_name == 'fbvae':
                if base_kwargs['iters'] > 100:
                    suffix += '_long'
            
                print(suffix)
                kwargs = {}
                kwargs.update(test_kwargs[j])
                kwargs.update(base_kwargs)
                [kwargs.pop(k) for k in ['homoscedastic', 'homo_y_var', 'cutoff', 'it_epochs']]
                test_traj, test_oracle_samples, test_gt_samples, test_max = optimization_algs.fb_opt(np.copy(X_train), oracles, ground_truth, vae_0, **kwargs)
            else:
                if base_kwargs['iters'] > 100:
                    suffix += '_long'
                kwargs = {}
                kwargs.update(test_kwargs[j])
                kwargs.update(base_kwargs)
                test_traj, test_oracle_samples, test_gt_samples, test_max = optimization_algs.weighted_ml_opt(np.copy(X_train), oracles, ground_truth, vae_0, **kwargs)
            np.save('../results/%s_traj%s.npy' %(test_name, suffix), test_traj)
            np.save('../results/%s_oracle_samples%s.npy' % (test_name, suffix), test_oracle_samples)
            np.save('../results/%s_gt_samples%s.npy'%(test_name, suffix), test_gt_samples )

            with open('../results/%s_max%s.json'% (test_name, suffix), 'w') as outfile:
                json.dump(test_max, outfile)
                
                
def run_killoran(killoran=True):
    """Runs the GFP comparative tests on the Killoran (aka AM-VAE) optimization algorithm"""
    TRAIN_SIZE = 5000
    train_size_str = "%ik" % (TRAIN_SIZE/1000)
    for i in range(3):
        RANDOM_STATE = i+1
        print(RANDOM_STATE)
        num_models = [1, 5, 20][i]
        X_train, _, _  = util.get_experimental_X_y(random_state=RANDOM_STATE, train_size=TRAIN_SIZE)

        LD=20
        L = X_train.shape[1]
        
        vae_suffix = '_%s_%i' % (train_size_str, RANDOM_STATE)
        
        ground_truth = gfp_gp.SequenceGP(load=True, load_prefix="data/gfp_gp")
        loss = losses.neg_log_likelihood
        keras.utils.generic_utils.get_custom_objects().update({"neg_log_likelihood": loss})
        oracle_suffix = '_%s_%i_%i' % (train_size_str, num_models, RANDOM_STATE)
        
        sess = tf.Session(graph=tf.get_default_graph())
        K.set_session(sess)
        vae = util.build_vae(latent_dim=20,
                  n_tokens=20, 
                  seq_length=X_train.shape[1],
                  enc1_units=50)
        vae.encoder_.load_weights("../models/vae_0_encoder_weights%s.h5" % vae_suffix)
        vae.decoder_.load_weights("../models/vae_0_decoder_weights%s.h5"% vae_suffix)
        vae.vae_.load_weights("../models/vae_0_vae_weights%s.h5"% vae_suffix)
        
        oracles = [keras.models.load_model("../models/oracle_%i%s.h5" % (i, oracle_suffix)) for i in range(num_models)]
        if not killoran:
            results, test_max = optimization_algs.killoran_opt(X_train, vae, oracles, ground_truth,
                                                               steps=30000, epsilon1=1e-5, epsilon2=1.,  
                                                               noise_std=1e-5,
                                                               LD=20, verbose=False, adam=False)
            
            np.save("../results/mala_results_%s_%i.npy" % (train_size_str, RANDOM_STATE), results)
            suffix = "_%s_%i" % (train_size_str, RANDOM_STATE)
            with open('results/%s_max%s.json'% ('mala', suffix), 'w') as outfile:
                json.dump(test_max, outfile)
                
        else:
            results, test_max = optimization_algs.killoran_opt(X_train, vae, oracles, ground_truth,
                                             steps=10000, epsilon1=0., epsilon2=0.1,  
                                             noise_std=1e-6,
                                             LD=20, verbose=False, adam=True)
            np.save("../results/killoran_may_results_%s_%i.npy" % (train_size_str, RANDOM_STATE), results)
            suffix = "_%s_%i" % (train_size_str, RANDOM_STATE)
            with open('../results/%s_max%s.json'% ('killoran', suffix), 'w') as outfile:
                json.dump(test_max, outfile)
    
    
def run_gomez_bombarelli(constrained=True):
    """Runs the GFP comparative tests on the Gomez-Bombarelli optimization algorithm"""
    TRAIN_SIZE = 5000
    train_size_str = "%ik" % (TRAIN_SIZE/1000)
    for it in range(3):
        RANDOM_STATE = it + 1
        
        X_train, _, _  = util.get_experimental_X_y(random_state=RANDOM_STATE, train_size=TRAIN_SIZE)
        ground_truth = gfp_gp.SequenceGP(load=True, load_prefix="../data/gfp_gp")
        
        L = X_train.shape[1]
        LD=20
        gt_var=0.01
        pred_vae = util.build_pred_vae_model(latent_dim=LD, n_tokens=X_train.shape[2], 
                                    seq_length=L, enc1_units=50, pred_var=gt_var)
        suffix = "_%s_%i" % (train_size_str, RANDOM_STATE)
        
        pred_vae.encoder_.load_weights("../models/pred_vae_encoder_weights%s.h5" % suffix)
        pred_vae.decoder_.load_weights("../models/pred_vae_decoder_weights%s.h5" % suffix)
        pred_vae.predictor_.load_weights("../models/pred_vae_predictor_weights%s.h5" % suffix)
        pred_vae.vae_.load_weights("../models/pred_vae_vae_weights%s.h5" % suffix)
        if not constrained:
            suffix = "_unconstrained" + suffix
        bomb_results, test_max = optimization_algs.bombarelli_opt(X_train, pred_vae, ground_truth, total_it=1000, constrained=constrained)
        with open('../results/%s_max%s.json'% ('bombarelli', suffix), 'w') as outfile:
            json.dump(test_max, outfile)
            
            
if __name__ == "__main__":
    
    ### Run weighted ML and FBVAE ###
    train_experimental_vaes()
    train_experimental_oracles()
    for it in [0, 1, 2]:
        run_experimental_weighted_ml(0, repeats=3)
    
    ### Run Gomez-Bombarelli method ###
    train_experimental_pred_vaes()
    run_gomez_bombarelli(constrained=True)
    run_gomez_bombarelli(constrained=False)
    
    ### Run AM-VAE method ###
    run_killoran(killoran=True)
        
    ### Run CbAS Q parameter test ### 
    test_cbas_xq(qs=[0.5, 0.75, 0.95, 1])