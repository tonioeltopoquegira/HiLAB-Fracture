"""Generate topology-optimized designs using the repo's inverse-design routines.

Creates designs with both MMA and Pixel-LBFGS and saves datasets + configs
for later training of a representation model.

Usage examples:
  python scripts/generate_designs.py --output_dir outputs/designs --opt_steps 100

You can override selection options such as `volfrac`, `penal_start`,
`penal_end`, and `penal_power`.
"""
import json
import os
import datetime
import logging

import xarray as xr
import numpy as np
import tensorflow as tf

from neural_structural_optimization import problems
from neural_structural_optimization import topo_api
from neural_structural_optimization import topo_physics
from neural_structural_optimization import models
from neural_structural_optimization import train
from neural_structural_optimization import pipeline_utils


def save_args_json(args_dict, path):
  # Convert numpy arrays to lists for JSON serialization
  out = {}
  for k, v in args_dict.items():
    try:
      # numpy arrays and autograd arrays have 'tolist'
      out[k] = v.tolist() if hasattr(v, 'tolist') else v
    except Exception:
      out[k] = str(v)
  with open(path, 'w') as f:
    json.dump(out, f, indent=2)


def main():
    # Editable configuration dict — modify this block directly.
    CONFIG = {
        'output_dir': 'outputs/designs',
        'problem': 'mbb_beam_192x64_0.4', #'',
        'opt_steps': 200,
        'volfrac': 0.4,
        'penal_start': None,
        'penal_end': None,
        'penal_power': None,
        'seed': 0,
        # how many designs to create per method (different seeds)
        'n_designs_per_method': 2,
        # binarization threshold
        'binarize_threshold': 0.5,
        # if True, only save the left half of the design (mirrored symmetry)
        'save_half': True,
        # standard deviation of Gaussian noise to add to initial z (0 -> constant init)
        'random_init_std': 0.01,
    }

    cfg = CONFIG

    logging.basicConfig(level=logging.INFO)

    problem_name = cfg['problem']
    if problem_name not in problems.PROBLEMS_BY_NAME:
        raise KeyError(
                f'Unknown problem {problem_name}. Available: {list(problems.PROBLEMS_BY_NAME)[:10]}...')

    problem = problems.PROBLEMS_BY_NAME[problem_name]

    # Build args from topo_api; these will be passed to PixelModel/Environment
    topo_args = topo_api.specified_task(problem)

    # override user-specified options from dict
    if cfg['volfrac'] is not None:
        topo_args['volfrac'] = float(cfg['volfrac'])
    if cfg['penal_start'] is not None:
        topo_args['penal_start'] = float(cfg['penal_start'])
    if cfg['penal_end'] is not None:
        topo_args['penal_end'] = float(cfg['penal_end'])
    if cfg['penal_power'] is not None:
        topo_args['penal_power'] = float(cfg['penal_power'])
    topo_args['opt_steps'] = int(cfg['opt_steps'])

    # prepare output directory
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    outdir = os.path.join(cfg['output_dir'], f'{problem_name}-{ts}')
    os.makedirs(outdir, exist_ok=True)

    # save config
    config_path = os.path.join(outdir, 'config.json')
    save_args_json(topo_args, config_path)
    logging.info('Saved config to %s', config_path)

    # Run multiple seeds per method and collect images + arrays into shared folders
    images_dir = os.path.join(outdir, 'images')
    arrays_dir = os.path.join(outdir, 'arrays')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(arrays_dir, exist_ok=True)

    n = int(cfg.get('n_designs_per_method', 1))
    thresh = float(cfg.get('binarize_threshold', 0.5))

    methods = [
        ('mma', train.method_of_moving_asymptotes),
        ('oc', train.optimality_criteria),
        ('pixel_lbfgs', train.train_lbfgs),
    ]

    for method_name, runner in methods:
        logging.info('Generating %d designs for method %s', n, method_name)
        combined_path = os.path.join(arrays_dir, f'{method_name}_designs.npy')
        combined_bin_path = os.path.join(arrays_dir, f'{method_name}_designs_binary.npy')
        # load existing combined arrays if present
        if os.path.exists(combined_path):
            try:
                combined_arr = np.load(combined_path)
            except Exception:
                combined_arr = None
        else:
            combined_arr = None
        if os.path.exists(combined_bin_path):
            try:
                combined_bin = np.load(combined_bin_path)
            except Exception:
                combined_bin = None
        else:
            combined_bin = None

        for i in range(n):
            seed = int(cfg['seed']) + i
            logging.info('  seed=%d', seed)
            model = models.PixelModel(seed=seed, args=topo_args)
            # ensure variables are created before assigning
            try:
                model(None)
            except Exception:
                # some models build lazily inside the runner; ignore if build happens later
                pass
            # optionally randomize initial z so different seeds start differently
            rnd_std = float(cfg.get('random_init_std', 0.0))
            if rnd_std > 0:
                try:
                    base = model.z.numpy()
                    rng = np.random.RandomState(seed)
                    noise = rng.normal(scale=rnd_std, size=base.shape)
                    new_z = np.clip(base + noise, 0.0, 1.0)
                    model.z.assign(tf.cast(new_z, model.z.dtype))
                except Exception:
                    logging.exception('  failed to randomize initial z for seed %d', seed)

            try:
                ds = runner(model, cfg['opt_steps'])
                run_dir = os.path.join(outdir, f'{method_name}-seed{seed}')
                os.makedirs(run_dir, exist_ok=True)
                ds_path = os.path.join(run_dir, f'{method_name}.nc')
                try:
                    ds.to_netcdf(ds_path)
                    logging.info('  saved trace to %s', ds_path)
                except Exception:
                    logging.exception('  failed to save trace for seed %d', seed)

                da = ds['design']
                final_da = da.isel(step=-1) if 'step' in da.dims else da

                # choose whether to save/render only the left half (center -> far left)
                save_half = bool(cfg.get('save_half', False))
                if save_half:
                    nx = final_da.sizes['x']
                    half = (nx + 1) // 2  # include center column when odd
                    da_to_render = final_da.isel(x=slice(0, half))
                else:
                    da_to_render = final_da

                # save per-seed images (only the chosen half if save_half=True)
                img_path = os.path.join(images_dir, f'{method_name}_seed{seed}.png')
                if save_half:
                    pipeline_utils.image_from_array(da_to_render.data).save(img_path)
                else:
                    pipeline_utils.image_from_design(da_to_render, problem).save(img_path)
                img_bin_path = os.path.join(images_dir, f'{method_name}_seed{seed}_binary.png')

                arr = np.asarray(da_to_render.data)
                bin_arr = (arr >= thresh).astype(np.float32)

                # save per-seed arrays
                np.save(os.path.join(run_dir, f'{method_name}_final.npy'), arr)
                np.save(os.path.join(run_dir, f'{method_name}_final_binary.npy'), bin_arr)

                # update combined arrays
                if combined_arr is None:
                    combined_arr = arr[None, ...]
                else:
                    combined_arr = np.concatenate([combined_arr, arr[None, ...]], axis=0)
                if combined_bin is None:
                    combined_bin = bin_arr[None, ...]
                else:
                    combined_bin = np.concatenate([combined_bin, bin_arr[None, ...]], axis=0)

                # save binarized image (for the saved slice)
                if save_half:
                    pipeline_utils.image_from_array(bin_arr).save(img_bin_path)
                else:
                    binary_da = da_to_render.copy(deep=False)
                    binary_da.data = bin_arr
                    pipeline_utils.image_from_design(binary_da, problem).save(img_bin_path)

            except Exception:
                logging.exception('Failed running method %s seed %d', method_name, seed)

        # write combined arrays back to disk (appending behavior achieved by load+concat)
        try:
            if combined_arr is not None:
                np.save(combined_path, combined_arr)
            if combined_bin is not None:
                np.save(combined_bin_path, combined_bin)
            logging.info('Wrote combined arrays for %s to %s / %s', method_name, combined_path, combined_bin_path)
        except Exception:
            logging.exception('Failed to write combined arrays for %s', method_name)

    logging.info('All done. Outputs in %s', outdir)


if __name__ == '__main__':
  main()
