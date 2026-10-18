"""Inspect determined solver inputs in dense-oracle tests."""


def extension_designs(base, *, extra_enrich_cols=(), extra_covar_kernels=()):
    from kasm.suffstats import _select_jk_extension

    for kind, names in (("burden", extra_enrich_cols), ("covar", extra_covar_kernels)):
        for name in names:
            extension = getattr(base, kind + "_extensions")[name]
            payload = dict(vars(extension.payload))
            if kind == "burden":
                payload = {
                    key: payload[key]
                    for key in ("total_hsq_weights", "annot_hsq_weights", "annot_het")
                }
                payload.update(annotation=name, total_het=base.total_het)
            yield kind, name, _select_jk_extension(base, extension), payload
