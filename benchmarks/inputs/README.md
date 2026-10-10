# Benchmark inputs

Inputs of the model-size and Netlib benchmarks. The scripts in `benchmarks/` read them from here by
default (the Netlib directory can be overridden with `bench_netlib.py --dir`).

| Directory | Contents | Source |
|---|---|---|
| `bigg/` | e_coli_core, iYO844, iMM904, iJO1366, iML1515 and Recon3D in SBML | BiGG Models (King et al. 2016), `https://bigg.ucsd.edu/static/models/<id>.xml.gz` |
| `netlib/` | the 44 Netlib LP problems used in the paper, in MPS format | Netlib LP collection, `https://www.netlib.org/lp/data/<problem>`, decompressed with Netlib's `emps.c` (`https://www.netlib.org/lp/data/emps.c`) |

The 12-species gut community used by most benchmarks (gapseq models, Western diet and initial
abundances) is part of the µODE repository (`examples/gut_western`); point `GUT_DIR` to it. The LP
workloads of the solver comparison are recorded from it with `benchmarks/record_workload.py` and
`benchmarks/make_reference.py` (see `benchmarks/REPRODUCE.md`).

## SHA-256 checksums

```
4c770b042c3e6e26ab4668b2900f4cf112d0a5d88a09a3520126ce0bc1605ff4  ./README.md.new
f45f88f908a1ed1a262148bbf327d34f1d20a36383e68a67d6078d626a6774d8  ./bigg/Recon3D.xml.gz
182a4dbb2a5899ea7e71134cc99f2e409d382f1c32c02984e27fe34e5e34ca79  ./bigg/e_coli_core.xml.gz
9140aca37cd25f7fd6be56be0cb2ceae06a89d7b3c7dc7087d1c348f3ccd25f6  ./bigg/iJO1366.xml.gz
2555e0f7e55a8cb8e770b9bb29cdaeb5db171941c414e7a232ff2d8e0228e308  ./bigg/iML1515.xml.gz
a35498a941ecb004e0a810226f07c2f57b090b4cd7da7ce4958b9f0a66d99f59  ./bigg/iMM904.xml.gz
ae819c25b166b08da23cb0cd7343d10c3c6afe15e868dd0467531ac08d6c6e80  ./bigg/iYO844.xml.gz
fec81e24fa91bc545d97239b108b43e6034f37b4bf2455a3f8c179726b44d44c  ./netlib/adlittle.mps
fd3562804ff19382a9cd8bcb22ec81bffd24a4143a2290783831d8c64516a24b  ./netlib/afiro.mps
a9628559a665e6739dd9d18b00e32236eeed7833ede3d1e2384f78c3e39b82af  ./netlib/agg.mps
bf1e15697ecfdc1f2e207b7f42509b03cd7e33580b99821950ac24a262145ae0  ./netlib/agg2.mps
165982069c227d9ea8cd4256e14971145d7cb366395343991ed1343c284c0c43  ./netlib/agg3.mps
244420cb770da5a27d579b09d0cf07228cd3a7f9d6392516edbb6445eff1013d  ./netlib/bandm.mps
c8bb193f8af5dcff735a3b8db62077db93625e93cfb26e2b01f83ca3039cbf7d  ./netlib/blend.mps
ffac93daf066802070a85c2aeaa6f6e2d5421540a9a34889ddbc6aa8a8698772  ./netlib/bnl1.mps
3f67c9b14db80aa0f4cff29baccb450623eb8aa609e37dd03aeb936dc785d362  ./netlib/boeing1.mps
ac6bd9c8cb95e61ab78a8a9f33b8326830ad20a6be944e074387e5805da2a3ef  ./netlib/boeing2.mps
53b17390ddb1831ef8ec3834303e39a21ac598c2fe159698bba342ea13909ed6  ./netlib/brandy.mps
7200826e526a47d53fb57b2b45661675217ede75406945e2820a0a73d3b946c0  ./netlib/capri.mps
19dff2bcc00c24502424e931814633e0b654ec30bfa6b29c9c81c5244198099f  ./netlib/degen2.mps
bbcec15c6e75ebe51040315386388b83099ab578e9b84131dc0568227a9901bb  ./netlib/e226.mps
bad29069b427ee33aa2a7110f164a7f522d01c337e326eb60e279cf0c9752c8e  ./netlib/etamacro.mps
401a243cc72b056bed39baeac3697fb4f95865d9b7993d215c3674daae31eea5  ./netlib/finnis.mps
b6948cd21f7fd6aca21cb38c12ea743151aa05a6eb95fa9b954dbfba6518938b  ./netlib/fit1d.mps
1222d49c497db33e1c1d6a91019ceca2975ca34d5d3cd102eb220a26528cbc60  ./netlib/grow15.mps
7072888d957cf01949854eac5ce171710d8b674566f1f2695d9f44de7b8cb74d  ./netlib/grow7.mps
119a04d815f7db33d67c0647144703a348087397f4440b2d83f1458d9253e5ab  ./netlib/israel.mps
b8a3df09b7e76059cb1abd04b8a36c6bf9d9781e20a3d51b38c56f31f47606ea  ./netlib/kb2.mps
21438697b88d8e6be102e3f14b9cd525abee550c18051ad10a6f7dca3ba8dd54  ./netlib/lotfi.mps
f967ab6466e4e9ad1914eb40d07d912149b4c5a6005252e86ba181048c3ca59c  ./netlib/recipe.mps
771da8cb34bf412831f3eab58edbd40a1c0f2962c61e3b1578eedbbb67682fb6  ./netlib/sc105.mps
3fde67ec21ff86d763704d4c20a4e5f94b86f6b03217e5b235ab4a53015f63d6  ./netlib/sc205.mps
c4571004af37a0c7d49099f8fcc513885401784d32f712bf44e014e40348c521  ./netlib/sc50a.mps
15c9d96e1d518dddc197e8e107febce590d0ae94594c676ff0d135b5642c1be5  ./netlib/sc50b.mps
53bca778f79d1ba8a0aea8235b906edf5b60828fdbfcd54575b1d5cdc1057350  ./netlib/scagr25.mps
30cccee0e5adb619dc0f1ea257b350a3cc407c1212d814249ab2ab6218936bc3  ./netlib/scagr7.mps
10fed25568aee378ae9f043a77ac7b53ba838eac06ec14f6664c24e481716177  ./netlib/scfxm1.mps
bb9ff57e9f39ddac9f02c96406f664b74861c324d128c3d7abf49c857abe6722  ./netlib/scsd1.mps
537ec314ac602d0e00b587d618f89cdf298033ff062deb96c7043a6160caf9cf  ./netlib/scsd6.mps
e63c14bbbc69433770a41b2495966d7484de6aa13877f79cbf1d606a8a41eece  ./netlib/scsd8.mps
546e421f29d61481619b4f7bc91b7fe0f07dbec53356a3e04f1ee8fb9fc32b6e  ./netlib/sctap1.mps
1a0b1e57cbbe3e8b101b27531002709a25f122dbb533f8621fcaeb41bc7cabd3  ./netlib/sctap2.mps
e31eb15e886236ff331c29433bb930017793606c7c5e86feb6c4e9f41fae4dde  ./netlib/share1b.mps
b861a8d9711956007c0419f956d29e78f20d12aac5b2c65d1a85a88468423754  ./netlib/share2b.mps
a63422b903f5711333363f3278d66731518a1ce1983efb468e062ed0cecdeb4f  ./netlib/ship04l.mps
07ef030b3ca52791503659a85ae58f8077ef16700468eaea00ca5bd19eb01cd5  ./netlib/ship04s.mps
e65acd93865b9757cfd6f0fb8de0db1d264a7f9c9022ff244fdb88d8d29f2d38  ./netlib/ship08s.mps
b38d3bb4f2a7aef8666fbc17496169223ccd86f0b7996c71be190ac4eab033ea  ./netlib/stair.mps
2ac6efdd6a6eaad89f47ea71411f8e0fbfbb24a7f31f4446133f39a41ce6fb31  ./netlib/standata.mps
a7996944d096c3cd74ca7cbeeb77e75804bcaaaba17620dba86108555912b4c3  ./netlib/stocfor1.mps
ab568fd861228875310ba3eb9f1d9eec1d8444c0116df78fbbc1eda38b6b9c8a  ./netlib/vtp.base.mps
```
