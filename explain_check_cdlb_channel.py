"""
说明程序: 解释 `check_cdlb_channel.py` 的工作流程以及验证依据。

该脚本调用 `check_cdlb_compliance` 并逐步说明验证逻辑:

1. 从 3GPP TR 38.901 读取 CDL-B 的相对路径功率与时延表。
2. 构造单天线、单极化的 `PanelArray` 并使用 `CDL` 类生成大量 CDL-B 信道抽样。
3. 统计每条路径的平均功率与时延, 与参考表进行比较。
4. 若功率相对误差小于 3% 且时延误差小于 1e-4 秒, 则认为实现符合标准。

由于 CDL 模型是随机过程, 其统计量在大样本下应收敛到标准提供的均值, 
因此通过统计检验可以判断实现是否遵循 3GPP 定义。
"""

from check_cdlb_channel import check_cdlb_compliance


def main():
    ok, err_p, err_tau = check_cdlb_compliance()
    print("check_cdlb_channel.py 通过比较统计功率和时延与标准值的误差来验证 CDL-B 模型实现。")
    print(f"最大相对功率误差: {err_p:.2e}")
    print(f"最大绝对时延误差: {err_tau:.2e}")
    if ok:
        print("误差均在容差范围内, 说明生成的 CDL-B 信道符合 3GPP 参考模型。")
    else:
        print("误差超出容差, CDL-B 信道可能不符合 3GPP 模型。")


if __name__ == "__main__":
    main()
