from numba import uint
import numpy as np

from spikeinterface.widgets.utils import get_segment_durations, validate_segment_indices
from spikeinterface.widgets.utils_matplotlib import make_mpl_figure 

from .base import BaseWidget, to_attr
from spikeinterface.core.sortinganalyzer import SortingAnalyzer
from spikeinterface.core.basesorting import BaseSorting

class FiringRateMapWidget(BaseWidget):
    """
    TODO
    """
    def __init__(
            self,
            sorting_analyzer_or_sorting: SortingAnalyzer | BaseSorting,
            bin_duration: int | float | None = None,
            segment_index: int | list[int] | None = None,
            unit_ids: list[int] | None = None,
            time_range: list[float | int] | None = None,
            sort_by_depth: bool = False,
            cmap: str | None = "hot",
            interpolation: str | None = None,
            backend: str | None = "matplotlib",
            **backend_kwargs,
        ):
            sorting = self.ensure_sorting(sorting_analyzer_or_sorting)

            if isinstance(segment_index, int): segment_index = [segment_index]
            segment_index = validate_segment_indices(segment_index, sorting)
            segment_index.sort()

            if unit_ids is None: unit_ids = sorting.unit_ids

            if time_range is not None: assert len(time_range) == 2, "'time_range' should be a list with start and end time in seconds"

            durations = get_segment_durations(sorting, segment_index)
            segment_boundaries = np.cumsum(durations)
            cumulative_durations = np.concatenate(([0], segment_boundaries))


            tot_duration = durations[-1] if time_range is None else (time_range[1] - time_range[0]) 
            if bin_duration is None: bin_duration = tot_duration / 100 
            n_bins = int(tot_duration / bin_duration)
            if time_range is None: time_range = [0, cumulative_durations[-1]]

            spike_train_data = {unit_id: np.array([]) for unit_id in unit_ids}

            for seg_idx in segment_index:
                for unit_id in unit_ids:
                    # Get spikes for this segment and unit
                    spike_times = (
                        sorting.get_unit_spike_train_in_seconds(unit_id=unit_id, segment_index=seg_idx)
                    )

                    # Store data
                    adjusted_times = spike_times + cumulative_durations[seg_idx]
                    spike_train_data[unit_id] = np.concatenate((spike_train_data[unit_id], adjusted_times))

            unit_indices = list(range(len(unit_ids)))
            y_ticks= {"ticks": unit_indices, "labels": unit_ids}

            #TODO (tayheau): update data_plot
            data_plot = dict(
                spike_train_data=spike_train_data,
                durations=durations,
                bin_duration=bin_duration,
                n_bins=n_bins,
                time_range=time_range,
                unit_ids=unit_ids,
                cmap=cmap,
                y_ticks=y_ticks,
            )

            BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)



    def plot_matplotlib(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        time_range = dp.time_range

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)
        scatter_ax = self.axes.flatten()[0]

        image = []
    
        for u_id, train_s in dp.spike_train_data.items():
            mask = (train_s > time_range[0]) & (train_s <= time_range[1])
            selected_spikes = train_s[mask]
            counts, _ = np.histogram(selected_spikes, dp.n_bins, range=tuple(time_range))
            firing_rate = counts / dp.bin_duration
            mean = np.mean(firing_rate)
            std = np.std(firing_rate)

            # image = np.stack((image, counts), axis=0)
            image.append(firing_rate)

        imshow_kwargs = dict(
            cmap=dp.cmap,
            interpolation="gaussian",
            aspect="auto",
            origin="lower",
            extent=[time_range[0], time_range[1], 0, len(dp.unit_ids)],
            # vmin=vmin,
            # vmax=vmax
        )

        im = scatter_ax.imshow(image, **imshow_kwargs)

        scatter_ax.set_xlabel("Time [s]")
        scatter_ax.set_ylabel("Unit id")
        scatter_ax.set_yticks(**dp.y_ticks)

        self.figure.colorbar(im, ax=scatter_ax)
