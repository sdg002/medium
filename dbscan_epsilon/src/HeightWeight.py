import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import math

def create_height_weight_data():
    src_points=[]
    src_points.append([160.0,60.0])
    src_points.append([160.0,61.0])
    src_points.append([161.0,59.0])
    src_points.append([159.0,61.0])

    src_points.append([180.0,80.0])
    src_points.append([179.0,81.0])
    src_points.append([179.0,82.0])
    src_points.append([180.0,82.0])

    #add some outliers
    src_points.append([170.0,42.0])
    return src_points

def find_dbscan_clusters(data_points):
    scaler = MinMaxScaler()
    scaler.fit(data_points)
    max_weight_variation=1  #1 Kg
    max_height_variation=10 #10 cm

    normalized_zero_zero=scaler.transform([[0,0]])
    normalized_thresholds=scaler.transform([[max_height_variation,max_weight_variation]])

    normalized_height_epsilon=normalized_thresholds[0][0]-normalized_zero_zero[0][0]
    normalized_weight_epsilon=normalized_thresholds[0][1]-normalized_zero_zero[0][1]
    
    epsilon=math.sqrt(normalized_height_epsilon**2 + normalized_weight_epsilon**2)
    min_samples=2
    normalized_data_points=scaler.transform(data_points)

    db=DBSCAN(eps=epsilon, min_samples=min_samples)
    db.fit(normalized_data_points)
    return db.labels_,normalized_data_points

def display(data_points:[],normalized_ht_wt_data:[],cluster_and_noise_labels:[]):
    #Plot the original data points
    x_axis_limits=[100,200]
    y_axis_limits=[20,120]

    x=list(map(lambda p: p[0],data_points))
    y=list(map(lambda p: p[1],data_points))

    fig, axs = plt.subplots(2)
    plt.subplots_adjust(hspace=0.5)
    #fig.suptitle('Demo of DBSCAN')
    axs[0].scatter(x, y,s=10,c='blue')
    axs[0].set_title("Height Weight")
    axs[0].set_xlim(x_axis_limits)
    axs[0].set_ylim(y_axis_limits)
    axs[0].set_xlabel("Height")
    axs[0].set_ylabel("Weight")

    #Plot the clusters
    axs[1].set_title("Clusters ")
    colors=["red","orange", "blue"]
    tuples_of_data_points_class_labels=list(zip(x,y,cluster_and_noise_labels))

    unique_labels=set(cluster_and_noise_labels)
    for label in unique_labels:
        noisy_points=list(filter(lambda t: t[2] == label,tuples_of_data_points_class_labels))
        point_x = list(map(lambda  p:p[0],noisy_points))
        point_y = list(map(lambda  p:p[1],noisy_points))
        if (label == -1):
            axs[1].scatter(point_x, point_y, marker='x', label="Outlier")
        else:
            legend_label="Cluster %d" % (label)
            axs[1].scatter(point_x, point_y, marker='o', label=legend_label, c=colors[label])
    axs[1].legend(loc="upper left")
    axs[1].set_xlim(x_axis_limits)
    axs[1].set_ylim(y_axis_limits)
    axs[1].set_xlabel("Height")
    axs[1].set_ylabel("Weight")

    plt.show()
    pass

ht_wt_data=create_height_weight_data()
cluster_and_noise_labels, normalized_ht_wt_data=find_dbscan_clusters(ht_wt_data)
display(ht_wt_data,normalized_ht_wt_data,cluster_and_noise_labels)
