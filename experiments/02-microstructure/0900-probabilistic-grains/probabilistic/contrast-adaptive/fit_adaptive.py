"""Train-only correction of the normalized-intensity anchoring mismatch.

Frozen from a training-only diagnostic before either variant's outer scores were
opened. Hardness labels are never an argument to this measurement-model fit.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, logsumexp


def _em(model, images, iterations, config, start_iteration=0):
    from grain_measurement import _responsibility
    offsets = np.zeros(len(images))
    prevalence = np.full(len(images), model.prior_phase_mean)
    for iteration in range(iterations):
        num, den = np.zeros(2), np.zeros(2)
        local = []
        alpha = config["beta_concentration"]*model.prior_phase_mean
        beta = config["beta_concentration"]*(1.-model.prior_phase_mean)
        for k, im in enumerate(images):
            d, v = im.d, im.pixel_var
            b, pi = offsets[k], prevalence[k]
            prior_var = model.offset_var + im.shade_var
            var = v[:, None]+model.tau2
            for _ in range(config["local_em_iterations"]):
                q = _responsibility(d, v, model.mu, model.tau2, b, pi)
                resp = np.column_stack([1.-q, q])
                pi = float(np.clip((q.sum()+alpha)/(len(q)+alpha+beta), .01, .99))
                prec = resp/var
                post_var = 1./(prec.sum()+1./prior_var)
                b = float(np.clip(post_var*np.sum(prec*(d[:, None]-model.mu)), -.5, .5))
            offsets[k], prevalence[k] = b, pi
            q = _responsibility(d, v, model.mu, model.tau2, b, pi)
            resp = np.column_stack([1.-q, q])
            prec = resp/var/len(d)
            num += (prec*(d[:, None]-b)).sum(0)
            den += prec.sum(0)
            local.append((im, b, resp, post_var))
        # Only a weak location prior for the bright component. Contrast is learned.
        prior_precision = np.array([config["adaptive_bright_prior_pseudoimages"]/max(model.tau2[0], .04**2), 0.])
        dtotal = den+prior_precision
        new_mu = (num+prior_precision*np.array([.05, 0.]))/np.maximum(dtotal, 1e-8)
        gap = config["adaptive_min_gap"]
        if new_mu[1]-new_mu[0] < gap:
            bright = (dtotal[0]*new_mu[0]+dtotal[1]*(new_mu[1]-gap))/dtotal.sum()
            new_mu = np.array([bright, bright+gap])
        new_mu[0] = np.clip(new_mu[0], -.25, 1.5-gap)
        new_mu[1] = np.clip(new_mu[1], new_mu[0]+gap, 1.5)
        scatter, count = np.zeros(2), np.zeros(2)
        for im, b, resp, post_var in local:
            variance = (im.d[:, None]-b-new_mu)**2-im.pixel_var[:, None]
            scatter += (resp*variance/len(im.d)).sum(0)
            count += resp.sum(0)/len(im.d)
        new_tau2 = np.clip(scatter/np.maximum(count, 1e-8), .04**2, .40**2)
        new_offset_var = np.clip(np.mean([b*b+pv-im.shade_var for im, b, _, pv in local]), .015**2, .20**2)
        model.mu = .5*model.mu+.5*new_mu
        model.tau2 = .5*model.tau2+.5*new_tau2
        model.offset_var = float(.5*model.offset_var+.5*new_offset_var)
        model.prior_phase_mean = float(np.clip(np.mean(prevalence), .05, .95))
        model.fit_history.append({"iteration": start_iteration+iteration, "mu": model.mu.tolist(),
                                  "tau": np.sqrt(model.tau2).tolist(), "offset_sd": np.sqrt(model.offset_var),
                                  "phase_prior_mean": model.prior_phase_mean})
    return model


def _balanced_laplace_score(model, images, config):
    from posterior_multimode import find_modes
    a = config["beta_concentration"]*model.prior_phase_mean
    b = config["beta_concentration"]*(1.-model.prior_phase_mean)
    scores = []
    for im in images:
        modes, _ = find_modes(model, im)
        masses = []
        for mode in modes:
            nlp, _, _, hessian = model._posterior(im, mode["point"], hessian=True)
            eigen = np.maximum(np.linalg.eigvalsh(hessian), 1e-4)
            # Constants omitted by the likelihood cancel between starts.
            masses.append(-nlp+np.log(2*np.pi)-.5*np.log(eigen).sum())
        log_evidence = (logsumexp(masses)-.5*np.log(2*np.pi*(model.offset_var+im.shade_var))-betaln(a, b))
        scores.append(log_evidence/len(im.d))
    return float(np.mean(scores))


def fit_adaptive(cls, images, config):
    images = [im for im in images if not im.empty]
    if len(images) < 2:
        raise ValueError("Insufficient nonempty training images")
    models, scores = [], []
    for initial_mean in config["adaptive_initial_means"]:
        model = cls(np.array(initial_mean, float), np.array([.12**2, .15**2]), .08**2, .30,
                    [im.image_id for im in images], [])
        model = _em(model, images, config["adaptive_warm_iterations"], config)
        scores.append(_balanced_laplace_score(model, images, config))
        models.append(model)
    index = int(np.argmax(scores))
    model = models[index]
    model.fit_history.append({"initialization_selection": "training_intensity_only_no_hardness",
                              "balanced_laplace_scores": scores, "chosen_initial_means": config["adaptive_initial_means"][index]})
    return _em(model, images, config["adaptive_refine_iterations"], config,
               start_iteration=config["adaptive_warm_iterations"])
