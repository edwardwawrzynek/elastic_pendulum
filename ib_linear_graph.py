import numpy as np
import scipy.stats

def plot_trendline(ax, x, trend, color, style, r2 = None):
  trend_f = np.poly1d(trend)
  label = "y = {:.3g} * x + {:.3g}".format(trend[0], trend[1])
  if r2 != None:
    label += ", $R^2 = $ {:.3}".format(r2)
  ax.axline((x[0], trend_f(x[0])), slope=trend[0], color=color, label=label, linewidth=1, linestyle=style)

def plot_ib_linear_graph(ax, x, x_err, y, y_err, do_trendline = True, do_r2 = True):
  # calculate linear trendline
  trend = np.polyfit(x, y, 1)
  # calculate error trendlines
  i0 = x.argmin()
  i1 = x.argmax()

  trend_err0 = np.polyfit(
    np.array([x[i0], x[i1]]),
    np.array([y[i0] - y_err[i0], y[i1] + y_err[i1]]),
    1
  )
  trend_err1 = np.polyfit(
    np.array([x[i0], x[i1]]),
    np.array([y[i0] + y_err[i0], y[i1] - y_err[i1]]),
    1
  )
  
  # draw trendline
  if do_trendline:
    # calculate r^2 for main trendline
    _, _, r, _, _ = scipy.stats.linregress(x, y)
    if do_r2:
      plot_trendline(ax, x, trend, 'r', "-", r * r)
    else:
      plot_trendline(ax, x, trend, 'r', "-")
    plot_trendline(ax, x, trend_err0, 'tab:gray', "--")
    plot_trendline(ax, x, trend_err1, 'tab:gray', "--")

  # draw points
  ax.errorbar(x, y, yerr = y_err, xerr = x_err, fmt = 'bo', markersize = 4, elinewidth=0.5, capsize=2, ecolor="k")

  # show trendline uncertainy in legend
  if do_trendline:
    t0 = (trend_err0[0] + trend_err1[0]) / 2
    t0_r = abs(trend_err0[0] - trend_err1[0]) / 2
    t1 = (trend_err0[1] + trend_err1[1]) / 2
    t1_r = abs(trend_err0[1] - trend_err1[1]) / 2
    trend_err_label = "y = ({:.3g} $\pm$ {:.3g}) * x + ({:.2e} $\pm$ {:.3g})".format(t0, t0_r, t1, t1_r)

    ax.legend(title=trend_err_label)
  else:
   ax.legend()