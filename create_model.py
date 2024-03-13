import pathlib
from typing import Tuple

import numpy
import torch
from bioimageio.core.build_spec import build_model
from cellpose import models

model_names = [
    "cyto",
    "nuclei",
    "cyto2",
    "CP",
    "CPx",

    # "tissuenet",
    # "livecell",
    "TN1",
    "TN2",
    "TN3",
    "LC1",
    "LC2",
    "LC3",
    "LC4",
]

model_names_CCBYNC = [
    # "tissuenet",
    # "livecell",
    "TN1",
    "TN2",
    "TN3",
    "LC1",
    "LC2",
    "LC3",
    "LC4",
]


def get_pt_model(model_type: str):
    model = models.Cellpose(model_type=model_type)
    return model.cp.net


def add_test_data(model_path, pt_net) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Returns the paths to the input, output, and style files.
    """
    input_ = numpy.random.rand(1, 2, 256, 256).astype("float32")
    with torch.no_grad():
        output, style = pt_net(torch.from_numpy(input_))

    input_pth = model_path / "input.npy"
    numpy.save(input_pth, input_)

    vector_flow_pth = model_path / "output_vector_flow.npy"
    style_pth = model_path / "output_style.npy"

    numpy.save(vector_flow_pth, output.numpy())
    numpy.save(style_pth, style.numpy())
    return input_pth, vector_flow_pth, style_pth


def write_doc_md(model_path, model_name) -> pathlib.Path:
    """
    Returns the path to the doc.md file.
    """
    doc_pth = model_path / "doc.md"
    doc_pth.write_text(
        f"# Cellpose {model_name} model\n\n"
        + "Bare network weights with minimal metadata to make it run. "
        + "Postprocessing not included!"
    )
    return doc_pth


for model in model_names:
    print(f"processing '{model}'.")
    cellpose_pt_net = get_pt_model(model)
    model_path = pathlib.Path(f"bioimage-model-{model}")
    model_zip = model_path / f"cp-{model}.zip"
    model_path.mkdir(exist_ok=True)
    weights_file = model_path / "cellpose_cyto.pt"
    torch.save(cellpose_pt_net.state_dict(), weights_file)
    input_file, output_file, style_file = add_test_data(model_path, cellpose_pt_net)
    doc_file = write_doc_md(model_path, model)

    if "nuclei" in model:
        diam_mean = 17
    else:
        diam_mean = 30

    build_model(
        weight_uri=str(weights_file),
        weight_type="pytorch_state_dict",
        # the test input and output data as well as the description of the tensors
        # these are passed as list because we support multiple inputs / outputs per model
        input_names=["input_image"],
        test_inputs=[input_file],
        input_min_shape=[[1, 2, 512, 512]],
        input_step=[[0, 0, 128, 128]],
        output_names=["vector_flow"],
        test_outputs=[
            output_file,
        ],
        input_axes=["bcyx"],
        output_axes=["bcyx"],
        output_offset=[[0, 0.5, 0, 0]],
        output_scale=[[1, 1, 1, 1]],
        halo=[[0, 0, 128, 128]],
        output_reference=["input_image"],
        architecture="./arch.py:CPnet2",
        model_kwargs={"nbase": [2, 32, 64, 128, 256], "sz": 3, "nout": 3},
        # where to save the model zip, how to call the model and a short description of it
        output_path=model_zip,
        name=f"cellpose-{model}",
        description=f"Cellpose '{model}' model",
        documentation=doc_file,
        # additional metadata about authors, licenses, citation etc.
        authors=[{"name": "Carsen Stringer", "affiliation": "Janelia"}],
        if model in model_names_CCBYNC:
            license="CC-BY-NC-4.0",
        else:
            license="CC-BY-4.0",
        tags=["cellpose-segmentation"],  # the tags are used to make models more findable on the website
        cite=[{"text": "Stringer et al.", "doi": "doi:10.1038/s41592-020-01018-x"}],
    )
