"""Hugging Face Accelerate prototype for the DS-A0 bounded spike.

Named ``accelerate_ddp`` rather than ``accelerate`` deliberately: a package
literally named ``accelerate`` under a directory that can reach ``sys.path``
risks shadowing the real distribution. The name also mirrors the sibling
``native_ddp`` spike so the two trees read as a matched pair, which is what
lets DG0 compare them.
"""
