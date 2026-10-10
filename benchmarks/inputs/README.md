# Benchmark inputs

Every input the paper's benchmarks read, so that the repository is self-contained. The
scripts in `benchmarks/` use these paths by default (the gut community directory can be
overridden with `GUT_DIR`, the Netlib directory with `bench_netlib.py --dir`).

| Directory | Contents | Source |
|---|---|---|
| `gut_western/` | 12 gapseq genome-scale models of human gut species (`gems/*.xml.gz`, ModelSEED namespace), the Western diet in the same namespace (`gems/western_gut_modelseed.csv`) and the initial relative abundances (`abundance.tsv`) | Reconstructed with gapseq (Zimmermann et al. 2021) from reference genomes; see `gut_western/gems/PROVENANCE.md`. The diet translates the VMH/AGORA Western gut diet as distributed with MICOM (Diener et al. 2020) to ModelSEED identifiers. |
| `bigg/` | e_coli_core, iYO844, iMM904, iJO1366, iML1515 and Recon3D in SBML | BiGG Models (King et al. 2016), `https://bigg.ucsd.edu/static/models/<id>.xml.gz` |
| `netlib/` | the 44 Netlib LP problems used in the paper, in MPS format | Netlib LP collection, `https://www.netlib.org/lp/data/<problem>`, decompressed with Netlib's `emps.c` (`https://www.netlib.org/lp/data/emps.c`) |
| `workloads/` | the recorded LP workloads and their certified reference answers (xz-compressed) | Recorded with `benchmarks/record_workload.py` and `benchmarks/make_reference.py` from the gut community above |

Decompress the workloads into `results/` before running the solver comparison:

```bash
mkdir -p results/solvers
xz -dc benchmarks/inputs/workloads/workload.pkl.xz > results/workload.pkl
xz -dc benchmarks/inputs/workloads/workload_coherent.pkl.xz > results/workload_coherent.pkl
xz -dc benchmarks/inputs/workloads/workload_reference.npz.xz > results/solvers/reference.npz
xz -dc benchmarks/inputs/workloads/workload_coherent_reference.npz.xz > results/workload_coherent_reference.npz
```

## SHA-256 checksums

```
f45f88f908a1ed1a262148bbf327d34f1d20a36383e68a67d6078d626a6774d8  ./bigg/Recon3D.xml.gz
182a4dbb2a5899ea7e71134cc99f2e409d382f1c32c02984e27fe34e5e34ca79  ./bigg/e_coli_core.xml.gz
9140aca37cd25f7fd6be56be0cb2ceae06a89d7b3c7dc7087d1c348f3ccd25f6  ./bigg/iJO1366.xml.gz
2555e0f7e55a8cb8e770b9bb29cdaeb5db171941c414e7a232ff2d8e0228e308  ./bigg/iML1515.xml.gz
a35498a941ecb004e0a810226f07c2f57b090b4cd7da7ce4958b9f0a66d99f59  ./bigg/iMM904.xml.gz
ae819c25b166b08da23cb0cd7343d10c3c6afe15e868dd0467531ac08d6c6e80  ./bigg/iYO844.xml.gz
a7bb56820cad44343c6392d5c6dd4d4ef2b4d3dcc3793d913bb23876285fe6a0  ./gut_western/abundance.tsv
7d617aee59da515b6c51b5d1c282a79d5405c42f5bc4f371c93ade7de5755894  ./gut_western/gems/A_muciniphila_BAA835.xml.gz
845fabfd43da9e3bf764cdf8201444514e8b24aa412caa8d72d35421e4b57783  ./gut_western/gems/B_fragilis_NCTC9343.xml.gz
90c584a435c432da2b25ed95873662625cdb042a94ca2293cd472d916f7a1630  ./gut_western/gems/B_longum_NCC2705.xml.gz
e849fd25f32fe9be27c80894d61b4f1c8b68f2ade786335a0f3bc58154a2856e  ./gut_western/gems/B_thetaiotaomicron_VPI5482.xml.gz
349ba99baf8511e739d9347a6ec09629bf04372cf011dffde5a83cd450e10d87  ./gut_western/gems/Bl_obeum_A2162.xml.gz
0de72ad36f348199280498e6faaa1959bd53ad8643be800a1a7e6ed0f7d3db7a  ./gut_western/gems/C_comes_ATCC27758.xml.gz
475890be65040cf05ac6e9c753c6f4d030f6fc5cce8b06c55d514156829a6dde  ./gut_western/gems/E_rectale_ATCC33656.xml.gz
9632d4705aa8832ad798a6cdcba14e5c3b9c282ac43914da5c69797638b48075  ./gut_western/gems/F_prausnitzii_A2165.xml.gz
8bcdef27481d562334783205bfa2b70e6d4312ffa81798ffd19fe6cd3a82d73e  ./gut_western/gems/L_acidophilus_NCFM.xml.gz
50641af1972607658bc02713266ab2eb2b0d9f5047d67207e14728448fa7ec3f  ./gut_western/gems/PROVENANCE.md
3bdcbe2ff1730d365a47ee24af4e5231bfa72184027f651bd89cc51c5d335036  ./gut_western/gems/P_copri_DSM18205.xml.gz
91ad89146e901d237035e3dfddf4a6efb4637723cac6c8a2bc881a5dec39edf6  ./gut_western/gems/R_bromii_L263.xml.gz
006156894c8a43fbde78b0c695add98782e640c05049c424612ef15d2e609d90  ./gut_western/gems/R_intestinalis_L182.xml.gz
31b5cbd4aa795337517fb735c37a9ce6415e857602cfdf4d06fd97e21923c029  ./gut_western/gems/western_gut_modelseed.csv
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
14b7f190c5020710c4933d877b941dd2c4973ffcdb82c565c8b3adcb7c13db40  ./workloads/workload.pkl.xz
f07e9d0185cbbd9b042f92cbf15250bcd031d69dc3618114ded9a4afcb91d5b8  ./workloads/workload_coherent.pkl.xz
9ab450c17aa6c3a4f098bd48f7977a47337c5fb219f8dfd383a2f08c602e7fd3  ./workloads/workload_coherent_reference.npz.xz
570d4eeb875a4e7d7f9a3bf18382200b850e2e3efd6ab102db9e7a9a3106c1c2  ./workloads/workload_reference.npz.xz
```
