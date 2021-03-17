import io
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

def visualize_dataset(dataset):
    X = np.array([dataset[i]["features"].tolist() for i in range(len(dataset))])
    y = np.array([dataset[i]["targets"].tolist() for i in range(len(dataset))])
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='jet')
    return fig


# def vis_dataset(dataset):
#     majority = np.array([dataset[i][0].tolist() for i in range(len(dataset)) if dataset[i][1] == 0.])
#     minority = np.array([dataset[i][0].tolist() for i in range(len(dataset)) if dataset[i][1] == 1.])
#     plt.scatter(majority[:, 0], majority[:, 1], c='b')
#     plt.scatter(minority[:, 0], minority[:, 1], c='r', alpha=0.5)
#     plt.show()


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    return img


def vis_histo_data(data):
    """
        Visualizes data as histogram
    """
    lims = (-5, 5)
    hist = np.histogram2d(data[:, 1], data[:, 0], bins=100, range=[lims, lims])
    plt.pcolormesh(hist[1], hist[2], hist[0], alpha=0.5)


def visualize_decision_boundary(X, y, model, plot_synthetic):
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    inputs = torch.tensor([list(xrow) for xrow in Xmesh]).float()
    scores = torch.sigmoid(model(inputs))
    Z = np.array([s.data > 0.5 for s in scores])
    Z = Z.reshape(xx.shape)
    Syn = None
    if plot_synthetic:
        Syn = model.z.detach().numpy()

    fig = plt.figure(figsize=(10, 10))
    plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    if Syn is not None:
        plt.scatter(Syn[:, 0], Syn[:, 1], c='g', s=40)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    return fig
