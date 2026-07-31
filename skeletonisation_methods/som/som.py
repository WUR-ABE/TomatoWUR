import minisom
import numpy as np


def som_method(points_filtered, cfg={}):
    # from plant_registration_4d import somSkeleton

    if  cfg.get("init") is None:
        size_som = 5
        som = minisom.MiniSom(size_som, size_som,  3, sigma=1, learning_rate=0.5, random_seed=1,
            # neighborhood_function="gaussian",
            # neighborhood_function="mexican_hat", # terrible
            neighborhood_function="bubble",
            # neighborhood_function="triangle",
            topology="rectangular",
            # topology="hexagonal",
            activation_distance="euclidean", #all others do not work that good. 'euclidean', 'cosine', 'manhattan', 'chebyshev'
        )
    else:
        som = minisom.MiniSom(**cfg["init"])

    data = points_filtered.copy()
    data = (data - data.min(axis=0)) / (data.max(axis=0)-data.min(axis=0))
    if cfg.get("weights") =="pca":
        som.pca_weights_init(data)
    elif cfg.get("weights") =="fps":
        import open3d as o3d
        # Convert data to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        num_samples = som._weights.shape[0] * som._weights.shape[1]

        pcd_downsampled = pcd.farthest_point_down_sample(num_samples=num_samples)
        fps_points = np.asarray(pcd_downsampled.points)
        # Get indices of downsampled points in the original data
        tree = o3d.geometry.KDTreeFlann(pcd)
        fps_points_idx = []
        for pt in fps_points:
            [_, idx_nn, _] = tree.search_knn_vector_3d(pt, 1)
            fps_points_idx.append(idx_nn[0])

        assert som._weights.shape[1]==1, f"SOM weights with y dimension > 1 currently not supported"

        # Assert that the shapes match before assignment
        assert som._weights.shape[0]*som._weights.shape[1] == len(fps_points), f"SOM weights shape {som._weights.shape[:2]} does not match FPS points shape {fps_points.shape[:2]}"
        # Assign FPS points to SOM weights for initialization
        idx = 0
        for i in range(som._weights.shape[0]):
            for j in range(som._weights.shape[1]):
                som._weights[i, j] = fps_points[idx]
                idx += 1
    else:
        som.random_weights_init(data)

    ## visualisation of initial weights
    # from scripts import visualize_examples as ve
    # ve.vis(pc=data, nodes = som._weights.reshape(som._weights.shape[0]*som._weights.shape[1],3))

    if cfg.get("train") is None:
        som.train(data, num_iteration=5, random_order=True, verbose=True, use_epochs=True)
    else:
        som.train(data, **cfg["train"])

    ## calculate average using win_map or
    return_indices = True
    winmap = som.win_map(data, return_indices=return_indices)
    nodes = []
    components=[]
    for w in winmap:
        if return_indices:
            nodes.append(data[winmap[w]].mean(axis=0))
            components.append(winmap[w])
        else:
        
            w_array = np.asarray(winmap[w])
            w_mean = np.mean(w_array, axis=0)
        # w_mean = np.median(w_array, axis=0)

            nodes.append(w_mean)
    nodes = np.array(nodes)
    
    # calculate weights directly not by averaging.
    # nodes = som._weights.reshape(-1,3)

    ## debugging differences between calculating nodes directly or using win_map
    debug_dif = False
    if debug_dif:
        weights_to_use = np.array([np.array([x[0], x[1]]) for x in winmap.keys()])
        nodes_weights = som._weights[weights_to_use[:,0],weights_to_use[:,1]].reshape(-1,3)
        if not nodes_weights.shape == nodes.shape:
            print("not possible to debug length is different")
        from scipy.spatial.distance import cdist
        Y = cdist(nodes_weights, nodes)
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(Y)
        avg_dist = Y[row_ind, col_ind].mean()
        print(avg_dist)

    ## unnormalize
    nodes = nodes * (points_filtered.max(axis=0)-points_filtered.min(axis=0)) + points_filtered.min(axis=0)


    return nodes
    # print(skeleton.shape)
    # # skeleton = som.get_weights().reshape(-1, 3)
    # visualize_examples.vis(pc=data, nodes=skeleton)
    # visualize_examples.vis_components(pc=data, components=components)