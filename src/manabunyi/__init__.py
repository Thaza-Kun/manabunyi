import dataclasses
from datetime import time, timedelta
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors

from tqdm import tqdm

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

@dataclasses.dataclass
class Coordinate:
    x: float
    y: float
    t: time
    name: str

    def __eq__(self, other):
        if isinstance(other, Coordinate):
            return (
                (self.x == other.x)
                and (self.y == other.y)
                and (self.t == other.t)
                and (self.name == self.name)
            )
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__repr__())

    def delay_to(self, other: "Coordinate") -> float:
        return timedelta(
            hours=other.t.hour - self.t.hour,
            minutes=other.t.minute - self.t.minute,
            seconds=other.t.second - self.t.second,
        ).total_seconds()

    def displaced_from(self, other: "Coordinate") -> (float, float):
        x_diff = other.x - self.x
        y_diff = other.y - self.y
        magnitude = np.sqrt(x_diff**2 + y_diff**2)
        try:
            angle = np.arctan(y_diff / x_diff)
            if x_diff >= 0 and y_diff >= 0:
                angle = angle
            elif x_diff < 0 and y_diff >= 0:
                angle = np.pi + angle
            elif x_diff < 0 and y_diff < 0:
                angle = np.pi + angle
            elif x_diff >= 0 and y_diff < 0:
                angle = angle
            # print(x_diff, y_diff, angle)
        except ZeroDivisionError:
            print("ZeroDivisionError!")
            if y_diff >= 0:
                angle = -np.pi / 2
            elif y_diff < 0:
                angle = np.pi / 2
        except ValueError:
            pass
        return magnitude, angle

    def as_tuple(self) -> (float, float):
        return self.x, self.y

def delta_time(
    x,
    y,
    reference: Coordinate,
    other: Coordinate,
    speed_of_sound: float = 300,
    scale: float = 1.0,
) -> float:
    """Given the coordinates of `reference` and `other`, return the time delay between the two coordinates if the source of sound is located at x,y
    """
    displace_from_reference, _ = reference.displaced_from(Coordinate(x, y, t=time(), name="origin"))
    displace_from_other, _ = other.displaced_from(Coordinate(x, y, t=time(), name="origin"))
    return (displace_from_other - displace_from_reference) * scale / speed_of_sound


def gauss(x, A: float = 1, mu: float = 0, sigma: float = 1):
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

def pdf_map(
    field_time_delay: np.ndarray,
    observed_delay: float,
    bins: int | np.ndarray,
    tolerance: float = 0.05,
) -> np.ndarray:
    # We use abs time delay to build the histogram here 
    # because the field is symmetric and binning for non-abs 
    # will skew the histogram in a finite field especially at the edge
    hist, bins = np.histogram(np.abs(field_time_delay), density=True, bins=bins)
    hist = hist + np.flip(hist) / 2  # symmetric at t = 0
    likelihood = gauss(bins[:-1], A=1, mu=observed_delay, sigma=tolerance)
    prob = hist * np.diff(bins)[0] * likelihood
    # return prob assigned at each point in field_time_delay
    ret = prob[np.digitize(field_time_delay, bins[:-1], right=True) - 1]
    return ret


def main() -> None:
    meter_per_degree = 111_111.11
    coordinates = [
        Coordinate(
            y=4.652009000000008,
            x=101.104996,
            t=time(hour=11, minute=6, second=27),
            name="Taman Tasik Damai",
        ),
        Coordinate(
            y=4.661239000000004,
            x=101.161577,
            t=time(hour=11, minute=6, second=11),
            name="Taman Tanjung Perdana",
        ),
        Coordinate(
            y=4.686545000000004,
            x=101.11593299999997,
            t=time(hour=11, minute=6, second=27),
            name="Taman Klebang Putra",
        ),
        Coordinate(
            y=4.63705400000001,
            x=101.14411600000003,
            t=time(hour=11, minute=6, second=15),
            name="Taman Tambun Emas",
        ),
        Coordinate(
            y=4.6384689999999935,
            x=101.14616700000002,
            t=time(hour=11, minute=6, second=11),
            name="Ulu Kinta",
        ),
        Coordinate(
            y=4.6523620000000045,
            x=101.16058,
            t=time(hour=11, minute=6, second=11),
            name="Kg Melayu Batu 8",
        ),
        Coordinate(
            y=4.638680000000013,
            x=101.16154599999999,
            t=time(hour=11, minute=6, second=11),
            name="Masjid Permai The Haven",
        ),
        Coordinate(
            y=4.609346999999994,
            x=101.10359400000002,
            t=time(hour=11, minute=6, second=32),
            name="Stadium Perak",
        ),
    ]
    x = np.linspace(101.10, 101.17, 1000)
    y = np.linspace(4.6, 4.7, 1000)
    X, Y = np.meshgrid(x, y)
    ERROR = 2
    bins = np.linspace(-30, 30, 10_000)

    Ps = []
    for n in range(len(coordinates)):
        PDFs = []
        origin = coordinates[n - 1]
        f, axs = plt.subplots(2, 1, height_ratios=[10, 1])
        axs[0].scatter(origin.x, origin.y, zorder=5, label=origin.t)
        
        for i, other in enumerate(tqdm(set(coordinates) - {origin})):
            delay = origin.delay_to(other)
            T = delta_time(X, Y, origin, other, scale=meter_per_degree)
            P = pdf_map(T, delay, bins=bins, tolerance=ERROR)
            PDFs.append(P)
            axs[0].scatter(other.x, other.y, zorder=5, label=other.t)
            P = reduce(lambda p0, p1: p0 * p1, PDFs)
            CS = axs[0].contourf(X, Y, P, levels=10)
            axs[0].set_xticks(np.arange(x.min(), x.max(), 0.02))
            axs[0].set_yticks(np.arange(y.min(), y.max(), 0.02))
            axs[0].xaxis.set_major_formatter(LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True)) # set lons
            axs[0].yaxis.set_major_formatter(LatitudeFormatter(number_format='0.2f',degree_symbol='')) # set lats
            axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=10, title='Masa Kedengaran')

            norm = colors.Normalize(vmin=CS.zmin, vmax=CS.zmax)
            scalarmap = plt.cm.ScalarMappable(norm=norm, cmap=CS.cmap)
            scalarmap.set_array([])
            plt.colorbar(scalarmap, orientation='horizontal', cax=axs[1], ticks=CS.levels)
            axs[1].ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            plt.suptitle('Taburan Kebarangkalian Kedudukan Punca Bunyi')
            plt.tight_layout()
            plt.savefig(f"out/map_from_{n}_at_{i}.png")

        P = reduce(lambda p0, p1: p0 * p1, PDFs)
        P = P / np.sum(P)
        Ps.append(P)
        CS = axs[0].contourf(X, Y, P, levels=100)
        plt.colorbar(CS, orientation='horizontal', cax=axs[1])
        axs[1].ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.suptitle('Taburan Kebarangkalian Kedudukan Punca Bunyi')
        plt.tight_layout()
        plt.savefig(f"out/map_from_{n}_overall.png")
        plt.close()


    openstreetmap = cimgt.OSM()
    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1])
    ax0 = plt.subplot(gs[0], projection=openstreetmap.crs)
    ax1 = plt.subplot(gs[1])
    ax0.set_extent([x.min(), x.max(), y.min(), y.max()])
    ax0.add_image(openstreetmap, 18)

    P = reduce(lambda p0, p1: p0 + p1, Ps)/len(Ps)
    CS = ax0.contour(X, Y, P, transform=ccrs.PlateCarree(),zorder=5)
    ax0.set_xticks(np.arange(x.min(), x.max(), 0.02), crs=ccrs.PlateCarree())
    ax0.set_yticks(np.arange(y.min(), y.max(), 0.02), crs=ccrs.PlateCarree())
    ax0.xaxis.set_major_formatter(LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True)) # set lons
    ax0.yaxis.set_major_formatter(LatitudeFormatter(number_format='0.2f',degree_symbol='')) # set lats
    for coord in coordinates:
        ax0.scatter(coord.x, coord.y, zorder=5, transform=ccrs.PlateCarree(), label=coord.t)
    ax0.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=10, title='Masa Kedengaran')
    
    norm = colors.Normalize(vmin=CS.zmin, vmax=CS.zmax)
    scalarmap = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    scalarmap.set_array([])
    plt.suptitle('Taburan Kebarangkalian Kedudukan Punca Bunyi')
    plt.colorbar(scalarmap, orientation='horizontal', cax=ax1, ticks=CS.levels)
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig("out/map_overall_overlap.png")
    plt.close()