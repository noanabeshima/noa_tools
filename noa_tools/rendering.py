from PIL import Image
import torch
import numpy as np

# there's something wrong with importing this.
# from scipy import stats

import math
import einops
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt


def is_tensor(x):
    return isinstance(x, torch.Tensor) or isinstance(x, torch.nn.parameter.Parameter)


def tensor_to_numpy(x):
    # if x is a tensor, convert to numpy array
    if isinstance(x, torch.Tensor):
        if x.device != torch.device("cpu"):
            x = x.cpu()
        if x.dtype == torch.bool:
            x = x.int()
        if x.dtype in {torch.float16, torch.bfloat16, torch.float64}:
            x = x.float()
        x = x.detach().cpu().numpy()
    if isinstance(x, torch.nn.parameter.Parameter):
        x = tensor_to_numpy(x.data)
    return x


def render_array(arr, scale: int = 1, raw_array=False):
    arr = tensor_to_numpy(arr)
    assert isinstance(arr, np.ndarray)
    arr = np.squeeze(arr)
    assert len(arr.shape) == 2
    if scale != 1:
        arr = np.kron(arr, np.ones((scale, scale)))

    if np.issubdtype(arr.dtype, np.floating):
        if arr.min() < 0.0 or arr.max() > 1.0:
            if arr.min() > -1e-2 and arr.max() < 1.0 + 1e-2:
                arr = np.clip(arr, 0, 1)
            else:
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-2)
        arr = (255 * arr // 1).astype(np.uint8)
    else:
        assert np.issubdtype(arr.dtype, np.integer)

    if raw_array is True:
        return arr
    else:
        return Image.fromarray(arr)


def render_array_w_sign(arr, scale: int = 1):
    pos = np.clip(arr, 0, None)
    neg = np.clip(arr, None, 0).abs()
    pos = render_array(pos / arr.abs().max(), scale=scale, raw_array=True)
    neg = render_array(neg / arr.abs().max(), scale=scale, raw_array=True)

    both = np.zeros(pos.shape + (3,), dtype=np.uint8)
    both[..., 0] = neg
    both[..., 1] = pos

    return Image.fromarray(both)


def str_arr_add(*args):
    """
    Casts tensors/nparrays to numpy string arrays and adds all items together,
    casting to string and broadcasting if necessary
    """
    if len(args) == 0:
        return ""
    args = list(args)
    for i, item in enumerate(args):
        item = tensor_to_numpy(item)
        if isinstance(item, np.ndarray):
            args[i] = item.astype(str)
    res = args[0]
    for item in args[1:]:
        res = np.char.add(res, item)
    return res

float_arr_to_str = np.vectorize(lambda x: f'{x:.2g}')

def heatmap(
    arr,
    perm_0=None,
    perm_1=None,
    dim_names=("row", "col"),
    info_0=None,
    info_1=None,
    include_idx=(None, None),
    title=None,
    mask_0=None,
    mask_1=None,
    sort_0=None,
    sort_1=None,
    ticks_0=None,
    ticks_1=None,
    indexed_keys=False,
    width=None,
    height=None
):
    """
    arr: 2d numpy or torch array to render
    perm_0, perm_1: permutation arrays to reorder the rows/columns of the heatmap
    dim_names : (str, str), names of dim 0 and dim 1 respectively
    info_0, info_1 : dictionary of string keys to list of strings describing the indices of dim 0 and dim 1 respectively
    include_idx: tuple of booleans, whether to include index in the hover info for each dimension
    title: str or None, title of the heatmap
    mask_0, mask_1: boolean arrays to mask out certain rows/columns of the heatmap
    sort_0, sort_1: 1d arrays of indices to sort the rows and columns of the heatmap by
    ticks_0, ticks_1: custom tick labels for dim 0 and dim 1 respectively
    indexed_keys: bool, if True, keys in info_0 and info_1 will be postfixed with their index
    width: int or None, width of the resulting figure in pixels
    height: int or None, height of the resulting figure in pixels
    """

    assert not (
        perm_0 is not None and sort_0 is not None
    ), "Cannot provide both perm_0 and sort_0"
    assert not (
        perm_1 is not None and sort_1 is not None
    ), "Cannot provide both perm_1 and sort_1"

    # convert arr to numpy array
    arr = tensor_to_numpy(arr)
    assert isinstance(arr, np.ndarray)

    # if include_idx[i] is None, it's set to True iff info_i is not provided
    if include_idx[0] is None:
        include_idx = (info_0 is None, include_idx[1])
    if include_idx[1] is None:
        include_idx = (include_idx[0], info_1 is None)
    
    # Create default title if none is provided
    if title is None:
        if dim_names == ("row", "col"):
            title = f"{arr.shape}"
        else:
            title = f"({dim_names[0]}, {dim_names[1]})"

    # get permutations from sort arrays
    if sort_0 is not None:
        assert len(sort_0.shape) == 1
        perm_0 = torch.tensor(tensor_to_numpy(sort_0)).topk(k=len(sort_0)).indices
    if sort_1 is not None:
        assert len(sort_1.shape) == 1
        perm_1 = torch.tensor(tensor_to_numpy(sort_1)).topk(k=len(sort_1)).indices

    # if permutations are not provided, use the identity permutation
    perm_0 = np.arange(arr.shape[0]) if perm_0 is None else tensor_to_numpy(perm_0)
    perm_1 = np.arange(arr.shape[1]) if perm_1 is None else tensor_to_numpy(perm_1)

    def construct_dim_info(
        dim_info: dict, dim_len, perm, mask, indexed_keys
    ):
        result = np.array(['' for _ in range(dim_len)])
        index = np.arange(dim_len).astype('U2')

        dim_info = {} if dim_info is None else dim_info

        for k, v in dim_info.items():
            if is_tensor(v):
                dim_info[k] = tensor_to_numpy(v)
            else:
                if not isinstance(v, np.ndarray):
                    dim_info[k] = np.array(v)

        dim_info = {k: v[perm] for k, v in dim_info.items()}
        result = result[perm]
        index = index[perm]

        if mask is not None:
            mask = tensor_to_numpy(mask)
            for k, v in dim_info.items():
                dim_info[k] = v[mask[perm]]
            result = result[mask[perm]]
            index = index[mask[perm]]
        

        if indexed_keys:
            result = str_arr_add(
                result,
                *[str_arr_add(k, ' ', index, ": ", v, "<br>") for k, v in dim_info.items()]
            )
        else:
            result = str_arr_add(
                result,
                *[str_arr_add(k, ": ", v, "<br>") for k, v in dim_info.items()]
            )
        return result

    # Construct dim info for each dimension (0 and 1)
    info_0 = construct_dim_info(
        info_0, arr.shape[0], perm_0, mask_0, indexed_keys
    )
    info_1 = construct_dim_info(
        info_1, arr.shape[1], perm_1, mask_1, indexed_keys
    )
    
    # construct hovertext
    hovertext = np.char.add(np.array(info_0)[:,None], np.array(info_1))

    if include_idx[0]:
        index_0 = construct_dim_info(
            {dim_names[0]: np.arange(arr.shape[0])},
            dim_len=arr.shape[0],
            perm=perm_0,
            mask=mask_1
        )
        hovertext = np.char.add(hovertext, index_0[:,None])
    if include_idx[1]:
        index_1 = construct_dim_info(
            {dim_names[1]: np.arange(arr.shape[1])},
            dim_len=arr.shape[1],
            perm=perm_1,
            mask=mask_0
        )
        hovertext = np.char.add(hovertext, index_1[None])
    
    # apply masks and permutations
    arr = arr[perm_0][:, perm_1]
    if mask_0 is not None:
        arr = arr[mask_0[perm_0]]
    if mask_1 is not None:
        arr = arr[:, mask_1[perm_1]]
    
    # finally add the array values to the hovertext
    hovertext = str_arr_add(hovertext, 'val: ', float_arr_to_str(arr), '<br>')

    # Create the plotly.graph_objects figure
    layout = go.Layout(yaxis=dict(autorange="reversed"))

    fig = go.Figure(
        data=go.Heatmap(
            z=arr,
            hoverinfo='text',
            text=hovertext,
            colorscale="Viridis",
        ),
        layout=layout,
    )

    fig.update_layout(
        xaxis_title=f"{dim_names[1]} ({arr.shape[1]})",
        yaxis_title=f"{dim_names[0]} ({arr.shape[0]})",
        title=title,
    )

    # Apply custom tick labels if provided
    if ticks_0 is not None:
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(ticks_0))),
            ticktext=ticks_0,
            tickangle=0,
            automargin=True,
        )
    else:
        fig.update_yaxes(showticklabels=False)

    if ticks_1 is not None:
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(len(ticks_1))),
            ticktext=ticks_1,
            tickangle=-45,
            automargin=True
        )
    else:
        fig.update_xaxes(showticklabels=False)
    
    # Set height and width if provided
    if height is not None:
        fig.update_layout(height=height)
    if width is not None:
        fig.update_layout(width=width)


    return fig


# commented out as there's some issue with scipy.stats
# def qq_plot(x, dist="norm", sparams=(), hovertext=None):
#     x = x.squeeze()
#     assert len(x.shape) == 1
#     perm = x.topk(x.shape[0], largest=False).indices
#     hovertext = np.array(hovertext)[perm] if hovertext is not None else None
#     qq = stats.probplot(x[perm], dist=dist, sparams=sparams)
#     x = np.array([qq[0][0][0], qq[0][0][-1]])
#     fig = go.Figure()
#     fig.add_scatter(x=qq[0][0], y=qq[0][1], mode="markers", hovertext=hovertext)
#     fig.update_xaxes(title="theoretical quantiles")
#     fig.update_yaxes(title="actual quantiles")

#     fig.add_scatter(x=x, y=qq[1][1] + qq[1][0] * x, mode="lines")
#     fig.layout.update(showlegend=False)
#     fig.show()


def get_image_grid(images, width: int = -1, scale: int = 1):
    """images is a list of PIL.Image images"""

    assert scale >= 1

    def find_closest_factors(n):
        for i in range(int(np.sqrt(n).item()) + 1, 0, -1):
            if n % i == 0:
                return i, n // i

    image_array = np.stack([np.array(image) for image in images], axis=0)
    if width == -1:
        rows, columns = find_closest_factors(len(images))
    else:
        assert isinstance(width, int)
        rows = math.ceil(len(images) / width)
        columns = width
        empty_images_needed = rows * columns - len(images)

        # add the empty images
        image_array = np.concatenate(
            [
                image_array,
                np.zeros((empty_images_needed,) + image_array.shape[1:]).astype(
                    np.uint8
                ),
            ],
            axis=0,
        )

    image_arr = einops.rearrange(
        image_array, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=rows, b2=columns
    )
    if scale != 1:
        image_arr = np.kron(image_arr, np.ones((scale, scale, 1))).astype(np.uint8)
    image_grid = Image.fromarray(image_arr)
    return image_grid


def plthist(x, *args, **kwargs):
    x = tensor_to_numpy(x)
    plt.histogram(x, *args, **kwargs)
    plt.show()


def hist(x, info=True, mask=None, ignore_small=False, *args, **kwargs):
    x = tensor_to_numpy(x)
    if info is True:
        info = np.arange(len(x))

    if ignore_small:
        abs_x = np.abs(x)
        x_max = np.max(abs_x)
        small = abs_x < 0.01 * x_max
        if mask is None:
            mask = ~small
        else:
            new_mask = np.zeros_like(x, dtype=bool)
            new_mask[mask] = True
            mask = new_mask | ~small
            assert mask.dtype == bool, mask.dtype
            mask = mask | ~small

    if mask is not None:
        mask = tensor_to_numpy(mask)
        if info is not None:
            info = info[mask]
        x = x[mask]

    if info is not None:
        fig = px.histogram(x, marginal="rug", hover_name=info, *args, **kwargs)
    else:
        fig = px.histogram(x, *args, **kwargs)
    fig.update_layout(showlegend=False)

    return fig
