from matplotlib import colors as mcolors
import colorsys


def adjust_color(color, lightness_mult=1.0, saturation_mult=1.0, hue_shift=0.0):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    h = (h + hue_shift) % 1.0
    l = max(0, min(1, l * lightness_mult))
    s = max(0, min(1, s * saturation_mult))

    return colorsys.hls_to_rgb(h, l, s)

def blend(color, target, t):
    """Blend color toward target by fraction t in [0, 1]."""
    c = mcolors.to_rgb(color)
    tgt = mcolors.to_rgb(target)
    return tuple((1 - t) * a + t * b for a, b in zip(c, tgt))


def make_alternate_versions_lightness(base_colors: list[str], n_versions: int):
    """
    Return n_versions palette variants of base_colors.

    Version 0 is the original palette.
    Later versions alternate between blending toward white and black,
    with increasing strength.
    """
    if n_versions < 1:
        return []

    if n_versions == 1:
        return [base_colors.copy()]

    versions = [base_colors.copy()]

    n_alternates = n_versions - 1
    max_lighten = 0.5
    max_darken = 0.35

    # how many lightened vs darkened versions we need
    n_light = (n_alternates + 1) // 2
    n_dark = n_alternates // 2

    light_steps = (
        [max_lighten * (i + 1) / n_light for i in range(n_light)]
        if n_light else []
    )
    dark_steps = (
        [max_darken * (i + 1) / n_dark for i in range(n_dark)]
        if n_dark else []
    )

    li = di = 0
    for k in range(n_alternates):
        if k % 2 == 0:
            t = light_steps[li]
            versions.append([blend(c, 'white', t) for c in base_colors])
            li += 1
        else:
            t = dark_steps[di]
            versions.append([blend(c, 'black', t) for c in base_colors])
            di += 1

    return versions


def make_alternate_versions_hue_shift(base_colors, n_versions, shift=0.18):
    versions = []
    for i in range(n_versions):
        hue_shift = shift * i   # was 0.08; much more obvious
        version = [
            adjust_color(
                c,
                lightness_mult=1.0,
                saturation_mult=1.0,
                hue_shift=hue_shift,
            )
            for c in base_colors
        ]
        versions.append(version)
    return versions

# def make_alternate_versions_hue_shift(base_colors, n_versions):
#     versions = []
#     for i in range(n_versions):
#         # example strategy: slightly rotate hue and vary brightness
#         hue_shift = 0.08 * i
#         lightness_mult = 1.0 - 0.12 * i
#         saturation_mult = 1.0

#         version = [
#             adjust_color(c, lightness_mult=lightness_mult,
#                          saturation_mult=saturation_mult,
#                          hue_shift=hue_shift)
#             for c in base_colors
#         ]
#         versions.append(version)
#     return versions


def make_alternate_versions(base_colors, n_versions, method="hue_shift", **kwargs):
    if method == "hue_shift":
        return make_alternate_versions_hue_shift(base_colors, n_versions, **kwargs)
    if method == "lightness":
        return make_alternate_versions_lightness(base_colors, n_versions, **kwargs)
