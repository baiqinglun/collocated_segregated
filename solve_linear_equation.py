'''求解线性方程'''
import numpy as np
from fp import Fp
from post_process import PostProcessManager
from solve import SolveManager

def eqn_scalar_norm2(solve: SolveManager, dim, it_nl, ncx, ncy, ncz, old_t, t, var):
    '''eqn_scalar_norm2'''
    l2_max_u = Fp(0.0)
    if var == "temperature":
        l2_u = solve.l2_t
        l2_max_u = solve.l2_max_t

    l2_u = np.sqrt(np.mean((t - old_t) ** 2))  # 差值的平方的平均值

    l2_max_u = max(l2_u, l2_max_u)

    return (l2_u, l2_max_u)


def scalar_Pj(dim,solve:SolveManager,post:PostProcessManager,current_iter, relax_factor, ncx, ncy, ncz, t_coefficient,t, init_zero,mesh_coefficient):
    '''高斯求解'''
    solve_equation_step_count = solve.solve_equation_step_count
    save_residual_frequency = post.save_residual_frequency
    linear_equation_residual_filename = post.linear_equation_residual_filename
    iter_step_count = solve.iter_step_count
    solve_equation_tolerance = solve.solve_equation_tolerance
    output_folder = post.output_folder

    initial_norm = 0.0
    if init_zero:
        t.fill(Fp(0.0))

    t_old = np.zeros((ncx, ncy, ncz), dtype=Fp)
    for equation_current_iter in range(1, solve_equation_step_count + 1):
        t_old = t.copy()
        norm2 = 0.0
        for k in range(ncz):
            for j in range(ncy):
                for i in range(ncx):
                    if i == 0:
                        tw = 0.0
                    else:
                        tw = t_old[i - 1, j, k]
                    if i == ncx - 1:
                        te = 0.0
                    else:
                        te = t_old[i + 1, j, k]

                    if j == 0:
                        ts = 0.0
                    else:
                        ts = t_old[i, j - 1, k]
                    if j == ncy - 1:
                        tn = 0.0
                    else:
                        tn = t_old[i, j + 1, k]

                    if dim == 3:
                        if k == 0:
                            tb = 0.0
                        else:
                            tb = t_old[i, j, k - 1]
                        if k == ncz - 1:
                            tt = 0.0
                        else:
                            tt = t_old[i, j, k + 1]
                    # t_new = (...)为定义t_new的值，赋值给等号右侧表达式的结果，一个值
                    t_new = (\
                                - t_coefficient[i, j, k, mesh_coefficient.AE_ID.value] * te\
                                - t_coefficient[i, j, k, mesh_coefficient.AW_ID.value] * tw\
                                - t_coefficient[i, j, k, mesh_coefficient.AN_ID.value] * tn\
                                - t_coefficient[i, j, k, mesh_coefficient.AS_ID.value] * ts\
                                + t_coefficient[i, j, k, mesh_coefficient.ABRC_ID.value] \
                        )
                    if dim == 3:
                        t_new = t_new - t_coefficient[i, j, k, mesh_coefficient.AT_ID.value] * tt - t_coefficient[i, j, k, mesh_coefficient.aB_id.value] * tb
                    t_new = t_new / t_coefficient[i, j, k, mesh_coefficient.AP_ID.value]
                    du = relax_factor * (t_new - t_old[i, j, k])
                    t[i, j, k] = t_old[i, j, k] + du
                    norm2 = norm2 + du * du

        cell_count = ncx * ncy * ncz
        norm2 = np.sqrt(norm2 / Fp(cell_count))

        if equation_current_iter == 1:
            initial_norm = norm2 + 1.e-20
        max_norm = -1.e20
        max_norm = max(norm2, max_norm) + 1.e-20
        rel_norm = norm2 / max_norm

        if rel_norm < solve_equation_tolerance or equation_current_iter == solve_equation_step_count:
            solve.solve_equation_total_count = solve.solve_equation_total_count + equation_current_iter
            if current_iter % save_residual_frequency == 0 or current_iter == 1 or current_iter == iter_step_count:
                print('current_iter, equation_current_iter, total_linsol_iters, norm2, initial_norm, max_norm, rel_norm ', current_iter, equation_current_iter, solve.solve_equation_total_count, norm2,
                      initial_norm, max_norm, rel_norm)

                with open(file=f"{output_folder}/{linear_equation_residual_filename}", mode='a',encoding='utf-8') as linear_equation_residual_filename_id:
                    print(current_iter, equation_current_iter, solve.solve_equation_total_count, norm2, initial_norm, max_norm, rel_norm, file=linear_equation_residual_filename_id)
            break
    return

'''
def scalarGs(case: StructuredMesh, solve: Solve, post: PostProcess, dim, it_nl, niter, relax, ncx, ncy, ncz, ncoef, ct,
             t, initzero, res):
    # define local variables related to the case object
    id_aP = case.aP_id
    id_aE = case.aE_id
    id_aW = case.aW_id
    id_aN = case.aN_id
    id_aS = case.aS_id
    if dim == 3:
        id_aT = case.aT_id
        id_aB = case.aB_id
    id_bsrc = case.b_id

    nsteps = solve.n_iter_step

    # define local variables related to the post object
    res_freq = post.res_freq
    linsol_fname = post.linsol_file_name

    if initzero:
        t.fill(fp(0.0))

    max_norm = -1.e20

    # Begin linear solver iterations
    for it in range(1, niter + 1):

        # Initialize norm
        norm2 = 0.0

        for k in range(ncz):
            for j in range(ncy):
                for i in range(ncx):

                    if i == 0:
                        tw = 0.0
                    else:
                        tw = t[i - 1, j, k]
                    if i == ncx - 1:
                        te = 0.0
                    else:
                        te = t[i + 1, j, k]

                    if j == 0:
                        ts = 0.0
                    else:
                        ts = t[i, j - 1, k]
                    if j == ncy - 1:
                        tn = 0.0
                    else:
                        tn = t[i, j + 1, k]

                    if dim == 3:
                        if k == 0:
                            tb = 0.0
                        else:
                            tb = t[i, j, k - 1]
                        if k == ncz - 1:
                            tt = 0.0
                        else:
                            tt = t[i, j, k + 1]

                    t_new = ( \
                                - ct[i, j, k, id_aE] * te \
                                - ct[i, j, k, id_aW] * tw \
                                - ct[i, j, k, id_aN] * tn \
                                - ct[i, j, k, id_aS] * ts \
                                + ct[i, j, k, id_bsrc] \
                        )

                    if dim == 3:
                        t_new = t_new - ct[i, j, k, id_aT] * tt - ct[i, j, k, id_aB] * tb

                    t_new = t_new / ct[i, j, k, id_aP]

                    du = relax * (t_new - t[i, j, k])
                    t[i, j, k] = t[i, j, k] + du
                    norm2 = norm2 + du * du

        ncell = ncx * ncy * ncz
        norm2 = np.sqrt(norm2 / fp(ncell))

        if it == 1:
            initial_norm = norm2 + 1.e-20

        max_norm = max(norm2, max_norm) + 1.e-20

        rel_norm = norm2 / max_norm

        if rel_norm < res or it == niter:
            solve.total_linsol_iters = solve.total_linsol_iters + it
            if it_nl % res_freq == 0 or it_nl == 1 or it_nl == nsteps:
                print('it_nl, it, tot_it, norm2, init, max, rel ', it_nl, it, solve.total_linsol_iters, norm2,
                      initial_norm, max_norm, rel_norm)
                with open(linsol_fname, 'a') as linsol_fid:
                    print(it_nl, it, solve.total_linsol_iters, norm2, initial_norm, max_norm, rel_norm, file=linsol_fid)
            break

    return
'''