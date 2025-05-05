import jax
import math
import numpy as np
import jax.numpy as jnp
import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt

def simulate_internal(
    pomp_obj=None, rinit=None, rprocess=None, ys=None, theta=None, 
    time_vec=None, covars=None, Nsim=100, state_names=None, 
    key=jax.random.PRNGKey(123), format = "array"
):
    if state_names is None:
        print(
            "Error: Please provide a list type containing 10 string elements, " 
            "each of which is a hidden state, in the same order as in the "
            "rprocess."
        )
    # get elements
    if pomp_obj is not None:
        rinit = pomp_obj.rinit
        rproc = pomp_obj.rproc
        ys = pomp_obj.ys
        theta = pomp_obj.theta
        covars = pomp_obj.covars
        rprocess = jax.vmap(rproc, (0, None, 0, None))

    if pomp_obj is not None:
        if rinit is None or rprocess is None or ys is None or theta is None:
            print("Missing Argument Input(s).")

    J=Nsim
    ylen = len(ys)
    initial_state = rinit(theta, J, covars)

    particles = initial_state
    state_list = []
    state_list.append(particles)
    for i in range (ylen):
        key, *keys = jax.random.split(key, num=J + 1)
        keys = jnp.array(keys)
        particles = rprocess(particles, theta, keys, covars)
        state_list.append(particles)

    state_names = state_names

    if time_vec is None:
        time = np.arange(0, ylen + 1)
    elif time_vec is not None and len(time_vec) == ylen:
        time = np.insert(time_vec, 0, 0)
    else:
        print(
            "Error in time vector: 'time_vec' should have the same length as "
            "'ys'."
        )
    
    latent_states = state_list[0].shape[1]

    n_cols = 3
    n_rows = math.ceil(latent_states / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    state_array = np.array(state_list)  
    t_critical = stats.t.ppf(0.975, df = J-1)
    mean = np.mean(state_array, axis=1)                     
    std = np.std(state_array, axis=1, ddof=1)               
    margin = t_critical * std / np.sqrt(J)
    lower_CI = mean - margin # (T + 1) x num_states
    upper_CI = mean + margin

    for state in range(latent_states):
        #ax = axes[state // 3, state % 3]  
        ax = axes[state]
        for j in range(J):
            particle_values = [state_list[t][j, state] for t in range(ylen + 1)]
            ax.plot(time, particle_values, alpha=0.7) 
        
        ax.plot(time, mean[:, state], color='black', linewidth=2, label='Mean')
        ax.plot(
            time, upper_CI[:, state], color='black', linestyle='--', 
            linewidth=1, label='Upper'
        )
        ax.plot(
            time, lower_CI[:, state], color='black', linestyle='--', 
            linewidth=1, label='Lower'
        )
        ax.set_title(state_names[state], fontsize=14)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel(f"State Value: {state_names[state]}", fontsize=12)
        ax.grid(True)

    for i in range(latent_states, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    
    state_array_reshaped = state_array.reshape(-1, latent_states)  
    day_vec = np.repeat(time, J)  
    state_df = pd.DataFrame(state_array_reshaped, columns=state_names)
    state_df['Real Time'] = day_vec
    state_cols = state_df.columns.tolist()
    state_cols.remove('Real Time')
    state_df = state_df[['Real Time'] + state_cols]

    lower_CI_arr = np.array(lower_CI)
    upper_CI_arr = np.array(upper_CI)
    lower_CI_df = pd.DataFrame(lower_CI_arr, columns = state_names)
    upper_CI_df = pd.DataFrame(upper_CI_arr, columns = state_names)
    lower_CI_df['Real Time'] = time
    upper_CI_df['Real Time'] = time

    lower_CI_cols = lower_CI_df.columns.tolist()
    lower_CI_cols.remove('Real Time')
    lower_CI_df = lower_CI_df[['Real Time'] + lower_CI_cols]

    upper_CI_cols = lower_CI_df.columns.tolist()
    upper_CI_cols.remove('Real Time')
    upper_CI_df = lower_CI_df[['Real Time'] + lower_CI_cols]

    
    if format == "array":
        return state_array, lower_CI_arr, upper_CI_arr
    elif format == "data.frame":

        return state_df, lower_CI_df, upper_CI_df
    else:
        print(
            "Error: in simulate: 'format' should be one of “arrays”, "
            "“data.frame”"
        )

def simulate(
    pomp_obj=None, rinit=None, rprocess=None, ys=None, theta=None, 
    time_vec=None, covars=None, Nsim=100, state_names=None, 
    key=jax.random.PRNGKey(123), format = "array"
):
    """
    Simulates the evolution of a system over time using a Partially Observed 
    Markov Process (POMP) model. This function can either execute on a POMP 
    object or utilize the specified parameters directly to perform the 
    simulation.

    Args:
        pomp_obj (Pomp, optional): An instance of the POMP class. If provided, 
            the function will execute on this object to perform the simulation. 
            If not provided, the necessary model components must be provided 
            separately. Defaults to None.
        rinit (function, optional): Simulator for the initial-state 
            distribution. Defaults to None.
        rprocess (function, optional): Simulator for the process model. 
            Defaults to None.
        ys (array-like, optional): The measurement array. Defaults to None.
        theta (array-like, optional): Parameters involved in the POMP model. 
            Defaults to None.
        time_vec (array-like, optional): Observation times as a vector. Defaults
            to None.
        covars (array-like, optional): Covariates for the process, or None if 
            not applicable. Defaults to None.
        Nsim (int, optional): The number of simulations to perform. Defaults to 
            100.
        state_names (list of str, optional): A list containing the latent state 
            names in the same order as in rprocess. Defaults to None.
        key (jax.random.PRNGKey, optional): The random key for random number 
            generation. Defaults to jax.random.PRNGKey(123).
        format (str, optional): The format of the return value, either "array" 
            or "data.frame". Defaults to "array".

    Returns:
        tuple: Depending on the 'format' argument, returns either:
            - A tuple containing arrays of the simulated states, lower 
              confidence intervals, and upper confidence intervals.
            - DataFrames of the simulated states, lower confidence intervals, 
              and upper confidence intervals.

    Raises:
        ValueError: If 'format' is not one of "array" or "data.frame".
    """

    return simulate_internal(
        pomp_obj, rinit, rprocess, ys, theta, time_vec, covars, Nsim, 
        state_names, key, format
    )


    