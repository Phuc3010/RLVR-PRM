# export CUDA_VISIBLE_DEVICES=2,3

# benchmarks=(AIME24 Math-500 Olympiad-Bench)
benchmarks=(EvoLM Omega Polaris-hard Polaris)
# Polaris Polaris-hard)
# benchmarks=(Polaris-hard)
# datasets=()
# model_name=EvoLM-1B-160BT-MixedFW8FM42-100k
model_name=Qwen2.5-3B

for benchmark in "${benchmarks[@]}"; do
    python prm_grade.py --model_name ${model_name} --benchmark $benchmark
    # python judge_grade.py --model_name ${model_name} --benchmark $benchmark
    # python grade.py --model_name ${model_name} --benchmark $benchmark
    # for dataset in "${datasets[@]}"; do
    #     python prm_grade.py --model_name ${model_name}-${dataset}-GRPO-step300 --benchmark $benchmark

done
done
