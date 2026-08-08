# PyZebArdYolo

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21386270.svg)](https://doi.org/10.5281/zenodo.21386270)

PyZebArdYolo is a real-time acquisition unit for behavioral neuroscience: a
graphical application that couples a consumer webcam to YOLO11-based object
detection (via Ultralytics / OpenVINO) and an Arduino Uno R3, delivering
position-contingent visual stimulation (RGB LEDs) to adult zebrafish
(*Danio rerio*) in closed loop, fully offline (no internet, no dedicated
GPU). It runs live camera feeds or pre-recorded videos and is intended for
scientific research.

> **Not the same software as "DRerio LogAI".** DRerio LogAI is a separate,
> more advanced multi-aquarium tracking and statistical-reporting platform
> by the same authors, registered as a computer program with INPI (Brazil)
> under process **BR 51 2026 005215-7**, titular **Universidade Estadual
> Paulista "Júlio de Mesquita Filho" (UNESP)**. PyZebArdYolo is not covered
> by that registration and is released independently. See `NOTICE` §0.

> **Package name note.** The internal Python package is named `zebtrack` for
> legacy reasons. This is **unrelated to *ZebTrack***, the separate MATLAB
> tracker (Luchiari lab, UFRN) that appears only as a comparator in the
> validation study under [`validation/`](validation/).

## Installation

This project is managed with [Poetry](https://python-poetry.org/).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MarkSant/PyZebArdYolo.git
    cd PyZebArdYolo
    ```

2.  **Install Poetry:**
    Follow the official instructions at [python-poetry.org](https://python-poetry.org/docs/#installation) to install Poetry on your system.

3.  **Install dependencies:**
    Once Poetry is installed, run the following command in the project root to create a virtual environment and install the required dependencies:
    ```bash
    poetry install
    ```

## Usage

To run the application, use the following command from the project's root directory:

```bash
poetry run python -m zebtrack
```

This will launch the main graphical user interface.

## Configuration

`config.yaml` holds the versioned defaults. Settings that belong to one
bench rather than to the project — camera index, Arduino serial port —
should go in **`config.local.yaml`**, which is merged on top of
`config.yaml` at load time and is not tracked by git. Only the keys you
override need to be present:

```yaml
camera:
  index: 1
arduino:
  port: 'COM3'
```

This keeps the serial port from being committed and re-committed every
time the hardware moves between machines.

## Output files

Each live recording session writes into its own folder, named
`<group>_<subject>` inside the project directory:

| File | Contents |
| --- | --- |
| `<base>.mp4` | Recorded video, stamped with the **measured** camera rate |
| `1_ProcessingArea_<base>.csv` | The processing polygon, as used for this session |
| `2_AreasOfInterest_<base>.csv` | The ROI rectangles, as used for this session |
| `3_CoordMovimento_<base>.csv` | Detected bounding boxes per frame |
| `6_Latency_<base>.csv` | One row per Arduino trigger: capture, decision, send and ACK timestamps, plus the derived latency legs |
| `7_FrameLedger_<base>.csv` | One row per frame handed to the video writer, making the video-to-pipeline frame mapping exact |
| `8_LatencyMeta_<base>.json` | Session metadata: measured fps, trigger and drop counts, detector and ROI configuration |

> **Latency data recorded before v1.2.0 is not valid.** The timing columns
> written by earlier versions were an instrumentation artefact — the serial
> acknowledgement was read one command late and the end-to-end timestamp
> referred to the wrong frame. Optical validation showed the resulting log
> understating real latency by 2.8x. Sessions recorded before v1.2.0 should
> be re-measured, not reanalysed.

## Architecture

The application is designed with a separation of concerns, loosely following a Model-View-Controller (MVC) pattern.

```mermaid
graph TD
    subgraph "User Interface (View)"
        GUI["GUI (Tkinter)"]
    end

    subgraph "Core Logic (Controller & Model)"
        AppController
        ProjectManager
        Detector["Detector (Ultralytics/OpenVINO)"]
        Settings
    end

    subgraph "I/O Subsystem"
        FrameSource["FrameSource (Camera/Video)"]
        Recorder
        Arduino
    end

    GUI -- User Actions --> AppController
    AppController -- Updates --> GUI

    AppController -- Manages --> ProjectManager
    AppController -- Uses --> Settings
    AppController -- Controls --> Detector
    AppController -- Controls --> Recorder
    AppController -- Controls --> Arduino
    AppController -- Gets Frames --> FrameSource

    Detector -- Processes frames provided by --> AppController
```

*   **GUI**: The user interface, built with Tkinter.
*   **AppController**: The central component that handles user input from the GUI and coordinates all other components.
*   **ProjectManager**: Manages the creation, loading, and saving of project files and configurations.
*   **Detector**: Performs object detection on video frames using models from `ultralytics` or `OpenVINO`.
*   **FrameSource**: Provides video frames, either from a live camera feed or a video file.
*   **Recorder**: Handles the saving of output video and tracking data.
*   **Arduino**: Manages communication with an Arduino board for hardware I/O.
*   **Settings**: Loads and manages application settings from configuration files.

## Repository layout

Besides the control software (`src/zebtrack/`), this repository ships the
material needed to build and reproduce the apparatus described in the
hardware paper:

*   **`firmware/Program_Final.ino`** — Arduino Uno R3 firmware (serial LED
    state machine; pins D13–D10, 9600 baud).
*   **`best12.pt`** + **`openvino_model_cache/best12_openvino_model/`** — the
    trained YOLO11s weights (PyTorch and the exported OpenVINO IR).
*   **`config.yaml`** — camera, Arduino, detector and ROI configuration.
*   **`validation/`** — the tracking-fidelity validation dataset and analysis
    (raw annotations, paired coordinates, metrics, figures and the analysis
    scripts). See [`validation/README.md`](validation/README.md). Scope: the
    PyZebArdYolo apparatus only; the *DRerio LogAI* platform is validated
    separately in its own repository.
*   **`hardware/`** — hardware design files for the custom acrylic arena
    (CAD). *(To be added: STL mesh and the editable source; a dimensioned
    drawing is in the paper.)*

## Authors

*   Marco Antônio Sant'Ana Camargos — São Paulo State University (UNESP), Botucatu, Brazil — marco.sant@unesp.br
*   Percília Cardoso Giaquinto — São Paulo State University (UNESP), Botucatu, Brazil — percilia.giaquinto@unesp.br

## Citation

If you use this software in your research, please cite it — see [`CITATION.cff`](CITATION.cff).

## License

The authors' own code (and the Arduino firmware) is MIT-licensed. However,
the **combined, distributed application** bundles Ultralytics YOLO
(AGPL-3.0), which makes the effective license of the distributed work
**AGPL-3.0**. Trained weights and the training dataset carry their own
attribution requirements (CC BY 4.0), and hardware design files are
CERN-OHL-S v2.

See [`LICENSE`](LICENSE) for the MIT text and [`NOTICE`](NOTICE) for the
full breakdown (third-party licenses, dataset attribution, and what
"effective AGPL-3.0" means for redistribution).
