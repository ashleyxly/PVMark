import argparse
import os
import sys
import time


def _bootstrap_cuda_visible_devices() -> None:
    if "--cuda_visible_devices" not in sys.argv:
        return
    try:
        index = sys.argv.index("--cuda_visible_devices")
        value = sys.argv[index + 1]
    except IndexError as exc:
        raise ValueError("--cuda_visible_devices requires a value.") from exc
    os.environ["CUDA_VISIBLE_DEVICES"] = value


_bootstrap_cuda_visible_devices()

import numpy as np
import torch
import transformers

from synthid_text import detector_mean
from synthid_text import logits_processing
from synthid_text import synthid_mixin


DEFAULT_MODEL_PATH = os.environ.get("PVMark_GPT2_MODEL", "gpt2")
DEFAULT_NEWS_TEXT = """
U.S. electricity demand is climbing fast enough that utilities, regulators and large industrial customers are preparing for another round of record-breaking consumption over the next two years, with data centers and artificial intelligence workloads emerging as the most closely watched source of growth. Power companies that spent much of the last decade planning around flat or only modestly rising demand are now revising load forecasts, accelerating investments in generation and transmission, and rethinking how they manage reliability during periods of extreme heat or unusually high industrial activity. Analysts say the shift is significant because it is being driven by several forces at once: new server campuses built for AI training, manufacturing investment tied to federal industrial policy, population growth in parts of the South and West, and the slow but steady electrification of homes, transport and business operations.

The new wave of concern is not simply about whether annual power use will reach a fresh high, but about how quickly demand is appearing in places where the grid is already under strain. Utilities serving regions with a heavy concentration of data-center development have reported a surge in requests for new interconnections, with some developers seeking power on timelines that do not match the pace of transmission construction or the lead times for large turbines, transformers and switchgear. That mismatch has turned what might once have been a straightforward planning exercise into a broader debate over who should bear the cost of upgrades, how to prioritize projects with uncertain completion dates, and whether all announced AI-related facilities will ultimately be built at the scale currently proposed.

Government forecasters have added urgency to that debate by projecting that U.S. power consumption could exceed prior records in both 2026 and 2027. The expectation of back-to-back highs has become a talking point for utility executives and policymakers because it suggests that the rise is not a short-lived rebound tied to weather or temporary industrial restocking, but part of a more durable change in the demand outlook. Energy economists caution that projections can still move with economic growth, commodity prices and construction schedules, yet they also note that the volume of announced investment in computing infrastructure is now large enough that even partial completion could materially alter regional power balances. In practical terms, that means grid planners are being asked to prepare for stronger load growth while still accounting for the risk that some customers arrive late or consume less than their initial requests implied.

Data centers are central to the story because AI computing requires enormous amounts of electricity not only to run processors continuously, but also to cool dense clusters of specialized chips operating in large buildings that must remain online around the clock. Utilities and consultants say the newest generation of facilities can demand far more power than older server farms, particularly when campuses are designed to expand in phases and ultimately house multiple buildings on the same site. As a result, a single project can influence local grid investment decisions, change assumptions about substation sizing and backup power, and complicate the way planners forecast peak demand. Some utilities have started asking data-center developers for more detailed milestones, deposits or contractual commitments before advancing expensive upgrades, arguing that speculative requests could otherwise crowd out infrastructure needed by other customers.

The supply side of the equation remains just as important. Natural gas-fired generation continues to play a large role in utility planning because it can be dispatched when demand spikes and because many regions still rely on gas plants to support reliability when wind or solar output changes quickly. At the same time, utilities are adding renewable generation, battery storage and transmission upgrades as they respond to customer pressure for cleaner power and to state or federal policy incentives that can improve project economics. The result is a more complicated buildout than the old model of simply adding another large thermal plant near a growing load center. Instead, companies are trying to combine flexible gas generation, renewable projects, storage assets and upgraded transmission lines in ways that can satisfy both reliability needs and corporate or regulatory expectations on emissions and costs.

That balancing act has exposed bottlenecks throughout the power sector. Transmission development often takes years because of permitting, land access, environmental review and cost allocation disputes, while major equipment orders can face long wait times when global supply chains are tight. Developers and utilities alike say transformers have been particularly hard to source quickly, and that delays in one part of a project can ripple through construction schedules elsewhere. Regional transmission organizations and utility commissions are therefore under pressure to streamline some approvals without weakening oversight. The challenge for policymakers is that speed matters, but so do consumer protections: if utilities overbuild for projects that arrive late or never reach full scale, households and smaller businesses may ultimately pay for infrastructure whose benefits are concentrated among a narrower group of large users.

Regional differences are shaping the response. In Texas, rapid population growth, industrial development and a business-friendly environment for data-center construction have combined to create one of the clearest examples of how new load can challenge an already stressed system. In the Mid-Atlantic and parts of the Midwest, developers are drawn to existing fiber networks, proximity to population centers and established energy infrastructure, but they are also running into congestion and connection queues that limit how quickly new facilities can be energized. Southeastern states have attracted investment with lower land costs and expanding utility systems, yet they too face questions about how much gas generation, solar capacity and transmission reinforcement will be needed if AI demand grows as quickly as current announcements suggest. In western markets, water availability, wildfire risk and transmission constraints add another layer of complexity for large campuses that require secure and continuous operations.

Large industrial customers are watching these developments closely because they do not want their own expansion plans or electricity bills to become collateral damage in the rush to serve AI. Manufacturers, chemical producers and other energy-intensive businesses have warned regulators that utilities should avoid shifting the cost of specialized infrastructure onto existing customers unless there are strong protections in place. Some have argued for tighter rules on deposits, contract length and cost recovery for especially large loads, saying that the traditional framework for serving new customers may not be sufficient when a single facility can require the equivalent of a small city's electricity demand. Consumer advocates have made a similar case, contending that families and small businesses should not subsidize speculative projects that promise jobs and tax revenue but may revise their schedules as financing, chip supply and technology needs change.

Utilities counter that they cannot wait until every project is fully built before starting to prepare the grid. They say long-lived infrastructure requires early decisions, and that failing to plan ahead could create shortages, higher spot prices and reliability problems if the demand surge materializes faster than expected. Executives also argue that many of the same investments needed for data centers, including transmission upgrades and new generation, can deliver broader benefits by strengthening the grid for homes, hospitals, factories and future electrification. The policy question therefore is not whether to invest, but how to sequence those investments and structure contracts so that the cost and risk are shared fairly. Several utilities have signaled that they want more flexible tariff structures or bespoke service agreements for very large loads, rather than forcing every unusual project into rate designs built for a different era.

Technology companies, for their part, have tried to present themselves as partners rather than simply huge new buyers of electricity. Some firms are signing long-term power purchase agreements for wind, solar and battery projects, while others are exploring advanced geothermal, small modular nuclear reactors or cleaner forms of backup generation. Many have also promoted strategies such as demand response, load shifting and on-site generation to reduce stress on the grid during peak periods. Even so, utilities and regulators remain cautious. They note that voluntary commitments can be helpful but may not fully solve the operational challenge posed by data centers that expect near-continuous service and have little tolerance for interruption. In some regions, the willingness of operators to curtail load during emergencies will become a key test of whether the AI industry can integrate smoothly into existing power systems.

The economics of the buildout are also under scrutiny because capital costs remain high and interest rates, while lower than peak levels in some forecasts, still make large infrastructure programs expensive. Utilities must decide which projects are urgent enough to move now, which can wait for clearer customer commitments, and how to explain rising capital expenditure plans to regulators and investors. Power producers see opportunity in this environment because sustained load growth can improve the case for new plants and transmission investments, but they also know that public tolerance for sharply higher bills is limited. If fuel costs rise, construction schedules slip, or reliability events expose weak points in the system, the political debate could shift quickly from enthusiasm about economic development to criticism over who benefited most and who paid the price.

Another complication is that electricity demand from AI is arriving while parts of the grid are already adapting to more variable weather, the retirement of older thermal plants and the expansion of intermittent renewable resources. Planners are no longer optimizing for a system dominated by stable baseload generation and predictable demand growth. Instead, they are managing a grid where summer heat waves can push air-conditioning load sharply higher, where storms can disrupt fuel or transmission availability, and where a cluster of newly connected data centers can alter local demand patterns far faster than traditional forecasting models were designed to capture. That environment increases the value of flexible generation, storage, regional coordination and better forecasting, but it also increases the cost of mistakes.

Regulators are trying to keep pace by opening proceedings on large-load interconnection rules, cost allocation and utility forecasting assumptions. Some commissions want utilities to prove that projected demand from data centers is credible before approving major spending, while others are emphasizing the need to preserve reliability even if that means acting before every uncertainty is resolved. Grid operators are likewise examining whether current market rules send the right signals for new generation, transmission and flexible demand resources. The debate can become technical quickly, yet the underlying issue is easy to grasp: when a fast-growing industry arrives on a system that takes years to expand, institutions must decide how much risk to absorb in advance and how much to place on the customer seeking service.

For power producers, the boom recalls earlier periods when infrastructure investment surged because of a structural shift in the economy. Some executives compare it loosely to past expansions tied to suburban growth, the rise of air conditioning, or earlier industrial buildouts that changed expectations for fuel supply and generation needs. There are similarities in the scale of capital required and in the way utilities once again find themselves at the center of a broader economic story. But there are also important differences. Today's demand surge is arriving in a more contested policy environment, with sharper scrutiny of emissions, higher expectations around resilience, more complicated permitting, and a technology sector whose plans can evolve rapidly with chip design, software efficiency and competition among cloud providers.

Those uncertainties have not stopped companies from moving ahead. Across the country, utilities are updating integrated resource plans, independent power producers are evaluating new gas and renewable projects, and equipment suppliers are trying to expand capacity where they can. Some transmission developers hope the urgency around AI-related load will help build political support for projects that previously moved slowly, though they still face the familiar obstacles of siting disputes and fragmented authority. In boardrooms and statehouses alike, the calculation is similar: if the demand arrives, early movers may capture economic gains and avoid reliability failures; if the boom disappoints, they risk saddling customers and shareholders with expensive assets built for a growth profile that never fully materialized.

The next two years are therefore likely to be defined by negotiation as much as by construction. Utilities will negotiate with large customers over deposits, service terms and timelines. Regulators will negotiate, in effect, over how much speculative risk is acceptable in rates. Technology companies will negotiate for access to clean and reliable power at a pace that supports their computing ambitions. And consumers will demand reassurance that the system can absorb this new era of electricity use without sacrificing affordability. The outcome will shape not only who builds the next wave of energy infrastructure, but also how the costs and benefits of the AI buildout are distributed across the broader economy.

If the most bullish demand forecasts prove accurate, the U.S. power industry could enter a period of investment not seen in years, with consequences for fuel markets, transmission planning, utility finance and industrial competitiveness. Natural gas producers may benefit if dispatchable generation remains essential, renewable developers may gain if corporate buyers keep signing long-term contracts, and grid equipment manufacturers may see prolonged demand for hardware that has already become harder to source. But the most durable lesson may be institutional rather than technological: a grid built for slower-moving demand trends must now adapt to a sector capable of requesting enormous increments of power on compressed timelines. Whether that transition is remembered as a successful modernization effort or a costly scramble will depend on how effectively utilities, regulators, producers and technology companies align investment with realistic expectations.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SynthID watermark detection latency on a static long English news text."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Tokenizer path used to convert the news text into token IDs.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value. Set before torch init.",
    )
    parser.add_argument(
        "--token_lengths",
        type=int,
        nargs="+",
        default=[2000],
        help="Token lengths to benchmark after truncating the news text.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for detection.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of timed detection runs per token length.",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=0,
        help="Number of warmup detection runs per token length.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k used for SynthIDLogitsProcessor.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature used for SynthIDLogitsProcessor.",
    )
    parser.add_argument(
        "--score_type",
        choices=["mean", "weighted_mean", "both"],
        default="both",
        help="Which detector score computation to include in timing.",
    )
    parser.add_argument(
        "--cache_mode",
        choices=["cold", "warm"],
        default="cold",
        help="cold clears Rust/hash caches before each timed run; warm keeps caches hot.",
    )
    parser.add_argument(
        "--print_token_stats",
        action="store_true",
        help="Print effective token and ngram counts for each benchmark length.",
    )
    return parser.parse_args()


def clear_detection_caches() -> None:
    logits_processing.compute_keys_use_LCG_from_rustlib.cache_clear()
    logits_processing.invoke_sample_g_values_use_LCG_from_rustlib.cache_clear()
    logits_processing.compute_ngram_keys_use_LCG_from_rustlib.cache_clear()
    logits_processing.compute_keys_use_hash_from_rustlib.cache_clear()
    logits_processing.invoke_sample_g_values_use_hash_from_rustlib.cache_clear()
    logits_processing.compute_ngram_keys_use_hash_from_rustlib.cache_clear()


def build_detection_batch(
    tokenizer: transformers.PreTrainedTokenizer,
    news_text: str,
    batch_size: int,
    token_length: int,
    device: torch.device,
) -> tuple[torch.LongTensor, int]:
    full_token_ids = tokenizer(
        news_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    available_tokens = int(full_token_ids.shape[0])

    if token_length > available_tokens:
        raise ValueError(
            "Requested token length exceeds available news text length: "
            f"token_length={token_length}, available_tokens={available_tokens}. "
            "Use a smaller --token_lengths value or replace DEFAULT_NEWS_TEXT with a longer article."
        )

    token_ids = full_token_ids[:token_length].unsqueeze(0).repeat(batch_size, 1).to(device)
    return token_ids, available_tokens


def run_detection(
    token_ids: torch.LongTensor,
    logits_processor: logits_processing.SynthIDLogitsProcessor,
    eos_token_id: int,
    score_type: str,
):
    eos_token_mask = logits_processor.compute_eos_token_mask(
        input_ids=token_ids,
        eos_token_id=eos_token_id,
    )[:, logits_processor.ngram_len - 1 :]

    context_repetition_mask = logits_processor.compute_context_repetition_mask(
        input_ids=token_ids,
    )
    combined_mask = context_repetition_mask * eos_token_mask

    g_values = logits_processor.compute_g_values(
        input_ids=token_ids,
    )

    g_values_np = g_values.cpu().numpy()
    combined_mask_np = combined_mask.cpu().numpy()

    if score_type == "mean":
        return detector_mean.mean_score(g_values_np, combined_mask_np)
    if score_type == "weighted_mean":
        return detector_mean.weighted_mean_score(g_values_np, combined_mask_np)

    mean_scores = detector_mean.mean_score(g_values_np, combined_mask_np)
    weighted_mean_scores = detector_mean.weighted_mean_score(g_values_np, combined_mask_np)
    return mean_scores, weighted_mean_scores


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_detection(
    token_ids: torch.LongTensor,
    logits_processor: logits_processing.SynthIDLogitsProcessor,
    eos_token_id: int,
    warmup_runs: int,
    num_runs: int,
    score_type: str,
    cache_mode: str,
) -> dict:
    for _ in range(warmup_runs):
        if cache_mode == "cold":
            clear_detection_caches()
        run_detection(token_ids, logits_processor, eos_token_id, score_type)
    synchronize_if_needed(token_ids.device)

    timings_ms = []
    last_scores = None
    for _ in range(num_runs):
        if cache_mode == "cold":
            clear_detection_caches()
        synchronize_if_needed(token_ids.device)
        start = time.perf_counter()
        last_scores = run_detection(token_ids, logits_processor, eos_token_id, score_type)
        synchronize_if_needed(token_ids.device)
        end = time.perf_counter()
        timings_ms.append((end - start) * 1000)

    timings_ms = np.array(timings_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(timings_ms)),
        "std_ms": float(np.std(timings_ms)),
        "min_ms": float(np.min(timings_ms)),
        "p50_ms": float(np.percentile(timings_ms, 50)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "max_ms": float(np.max(timings_ms)),
        "last_scores": last_scores,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define eos_token for detection masking.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = dict(synthid_mixin.DEFAULT_WATERMARKING_CONFIG)
    config["device"] = device
    logits_processor = logits_processing.SynthIDLogitsProcessor(
        **config,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    available_tokens = int(
        tokenizer(
            DEFAULT_NEWS_TEXT,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].shape[0]
    )

    print(f"device: {device}")
    print(f"model_name_or_path: {args.model_name_or_path}")
    print(f"batch_size: {args.batch_size}")
    print(f"score_type: {args.score_type}")
    print(f"cache_mode: {args.cache_mode}")
    print(f"token_lengths: {args.token_lengths}")
    print(f"available_text_tokens: {available_tokens}")
    print("timing scope: eos mask + context repetition mask + g-value computation + detector score")
    print("excluded: tokenization, model loading, text generation")
    print("input source: static long English news text truncated to exact requested token length")
    print("-" * 100)

    with torch.no_grad():
        for token_length in args.token_lengths:
            if token_length < logits_processor.ngram_len:
                print(
                    f"skip token_length={token_length}: must be >= ngram_len ({logits_processor.ngram_len})"
                )
                continue

            token_ids, _ = build_detection_batch(
                tokenizer=tokenizer,
                news_text=DEFAULT_NEWS_TEXT,
                batch_size=args.batch_size,
                token_length=token_length,
                device=device,
            )

            num_input_tokens = token_ids.shape[1]
            num_scored_positions = num_input_tokens - (logits_processor.ngram_len - 1)
            if args.print_token_stats:
                print(
                    f"token_length={token_length} -> input_tokens={num_input_tokens}, "
                    f"scored_positions={num_scored_positions}, batch_size={args.batch_size}"
                )
                print(
                    "sample_text[0]: "
                    + tokenizer.decode(token_ids[0], skip_special_tokens=True)[:200].replace("\n", " ")
                )

            result = benchmark_detection(
                token_ids=token_ids,
                logits_processor=logits_processor,
                eos_token_id=tokenizer.eos_token_id,
                warmup_runs=args.warmup_runs,
                num_runs=args.num_runs,
                score_type=args.score_type,
                cache_mode=args.cache_mode,
            )

            print(
                "token_length={token_length:4d} | mean={mean_ms:8.3f} ms | "
                "p50={p50_ms:8.3f} ms | p95={p95_ms:8.3f} ms | "
                "min={min_ms:8.3f} ms | max={max_ms:8.3f} ms | std={std_ms:8.3f} ms".format(
                    token_length=token_length,
                    **result,
                )
            )
            print(f"last_scores: {result['last_scores']}")


if __name__ == "__main__":
    main()
