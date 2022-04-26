# Neural_filter (revised 04/06)

## To-do
1. Data construction
- Save channel taps
- Generate channel taps with fixed random seed
- ~~First, implement without noise (SNR should be considered later)~~

2. LMMSE equalizer
- Optimal filter
- LMMSE는 행렬 계산만 하면 됨.
- Test data에 대해서만 계산해봤는데, 결과가 나오지 않음.

(04/09) Sequence 순서가 거꾸로 된 것 같아서, sampled symbol을 거꾸로 잡아보는 실험 진행
- 현재 target 잡는 indexing이 맞음, e.g.) index 0 이 index L-1 로 사용되는 중임

(04/12) 상우형 feedback 따라서 간단한 channel 만들어보기
- Noise 안 넣고, channel tap 하나만 썼을 때 결과 잘 나옴
- Channel 크기와 noise 크기가 너무 차이나는거 아닐까?

- 그냥 decoding을 해보자! BER check (04/14)
- Noise 없이 + channel 간단하게 하면 BER 그냥 0 나옴
- Noise 섞는 순간 성능 망가짐 + W에 있는 SNR term 살려도 성능 안좋음
- Channel 복소수 성분 넣어봐도 (한 성분으로), 잘 되는데 이게 맞는 결과 혹은 식이 틀렸을듯
(04/17)
- Loss 수정하고 (2배), LMMSE가 경향성이 없는 것을 확인함
(04/26)
- Residual connection + Non-local block?
- GELU보다 ELU가 더 좋은데?