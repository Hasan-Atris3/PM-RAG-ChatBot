# # main2.py

# import os
# import time
# from dotenv import load_dotenv

# load_dotenv()

# from srd_engine_final import SRDChatbotEngine, ClaudeAnswerer


# def yes_no(prompt: str, default: bool = False) -> bool:
#     """
#     Simple [y/N] helper.
#     """
#     raw = input(prompt).strip().lower()
#     if not raw:
#         return default
#     return raw in ("y", "yes")


# def main():
#     engine = SRDChatbotEngine()
#     claude_text_llm = None  # Lazy init (for final answers)

#     print("\n" + "=" * 50)
#     print("      ULTIMATE SRD CO-PILOT (Claude + Qwen2-VL)")
#     print("=" * 50)

#     while True:
#         print("\n1. Index Documents")
#         print("2. Ask Question")
#         print("3. Exit")

#         choice = input("\nChoose: ").strip()

#         # ----- INDEX -----
#         if choice == "1":
#             pdf = input("Enter SRD PDF path: ").strip().strip('"')
#             if not os.path.exists(pdf):
#                 print("[ERROR] SRD PDF not found.")
#                 continue

#             gantt = (
#                 input("Gantt Chart path (optional): ").strip().strip('"') or None
#             )
#             cls = (
#                 input("Class Diagram path (optional): ").strip().strip('"') or None
#             )
#             seq = (
#                 input("Sequence Diagram path (optional): ").strip().strip('"')
#                 or None
#             )

#             # Vision choices
#             print("\nDiagram understanding options:")

#             # <<< ADDED: Ask user if they want Qwen2-VL >>>
#             use_qwen_vision = yes_no(
#                 "Use Qwen2-VL (free, open-source vision)? (y/N): ", default=False
#             )

#             # EXISTING CLAUDE OPTION
#             use_claude_vision = yes_no(
#                 "Also use Claude Vision for diagrams? (y/N): ", default=False
#             )

#             # User feedback
#             if use_qwen_vision and use_claude_vision:
#                 print("→ Diagrams will be processed by BOTH Qwen2-VL (free) and Claude Vision (paid).")
#             elif use_qwen_vision:
#                 print("→ Diagrams will be processed ONLY by Qwen2-VL (free).")
#             elif use_claude_vision:
#                 print("→ Diagrams will be processed ONLY by Claude Vision.")
#             else:
#                 print("→ No Vision AI selected. Using OCR only (fastest).")

#             try:
#                 engine.build_index(
#                     pdf_path=pdf,
#                     gantt_path=gantt,
#                     class_path=cls,
#                     seq_path=seq,
#                     use_qwen_vision=use_qwen_vision,     # <<< CHANGED FROM True
#                     use_claude_vision=use_claude_vision,
#                 )
#                 print("✔ Indexed successfully.")
#             except Exception as e:
#                 print(f"[ERROR] Indexing failed: {e}")

#         # ----- CHAT -----
#         elif choice == "2":
#             if not engine.vectorstore:
#                 print("Please index documents first (Option 1).")
#                 continue

#             if claude_text_llm is None:
#                 try:
#                     claude_text_llm = ClaudeAnswerer()
#                     print("✔ Claude (text) initialized for final answers.")
#                 except Exception as e:
#                     print(f"[ERROR] Failed to init Claude for answers: {e}")
#                     print(
#                         "Make sure 'anthropic' is installed and ANTHROPIC_API_KEY is set."
#                     )
#                     continue

#             while True:
#                 q = input("\n[You]: ").strip()
#                 if q.lower() in ("exit", "back", "quit"):
#                     break

#                 # <<< ADDED: Total question timer >>>
#                 total_start = time.time()

#                 # ----- Retrieval Timer -----
#                 retrieval_start = time.time()
#                 try:
#                     results = engine.hybrid_search(q, top_k=7)
#                 except Exception as e:
#                     print(f"[ERROR] Search failed: {e}")
#                     continue
#                 retrieval_time = time.time() - retrieval_start
#                 print(f"[Retrieved in {retrieval_time:.2f}s]")

#                 if not results:
#                     print("No matching content found in the SRD or diagrams.")
#                     continue

#                 # Debug: show where the info came from
#                 for r in results:
#                     src = r["metadata"].get("source")
#                     sect = r["metadata"].get("section", "N/A")
#                     score = r["score"]
#                     print(f" → {src} | section={sect} | score={score:.2f}")

#                 print("\n--- Claude Answer ---")

#                 # ----- Claude answering time -----
#                 claude_start = time.time()
#                 try:
#                     answer = claude_text_llm.generate_answer(q, results)
#                     print(answer)
#                 except Exception as e:
#                     print(f"Claude error: {e}")
#                 claude_time = time.time() - claude_start

#                 print(f"\n[Claude Answer Time: {claude_time:.2f}s]")

#                 total_time = time.time() - total_start
#                 print(f"[Total Time (Question → Final Answer): {total_time:.2f}s]")
#                 print("---------------------")

#         # ----- EXIT -----
#         elif choice == "3":
#             print("Goodbye.")
#             break

#         else:
#             print("Invalid choice. Please select 1, 2, or 3.")


# if __name__ == "__main__":
#     main()
# main2.py

import os
import time
from dotenv import load_dotenv

load_dotenv()

from srd_engine_final import SRDChatbotEngine, ClaudeAnswerer


def yes_no(prompt: str, default: bool = False) -> bool:
    """
    Simple [y/N] helper.
    """
    raw = input(prompt).strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def main():
    engine = SRDChatbotEngine()
    claude_text_llm = None  # Lazy init (for final answers)

    print("\n" + "=" * 50)
    print("      ULTIMATE SRD CO-PILOT (Claude + Qwen2-VL)")
    print("=" * 50)

    while True:
        print("\n1. Index Documents")
        print("2. Ask Question")
        print("3. Exit")

        choice = input("\nChoose: ").strip()

        # ----- INDEX -----
        if choice == "1":
            pdf = input("Enter SRD PDF path: ").strip().strip('"')
            if not os.path.exists(pdf):
                print("[ERROR] SRD PDF not found.")
                continue

            gantt = input("Gantt Chart path (optional): ").strip().strip('"') or None
            cls = input("Class Diagram path (optional): ").strip().strip('"') or None
            seq = input("Sequence Diagram path (optional): ").strip().strip('"') or None

            print("\nDiagram understanding options:")

            use_qwen_vision = yes_no(
                "Use Qwen2-VL (free, open-source vision)? (y/N): ", default=False
            )

            use_claude_vision = yes_no(
                "Also use Claude Vision for diagrams? (y/N): ", default=False
            )

            if use_qwen_vision and use_claude_vision:
                print("→ Diagrams will be processed by BOTH Qwen2-VL (free) and Claude Vision (paid).")
            elif use_qwen_vision:
                print("→ Diagrams will be processed ONLY by Qwen2-VL (free).")
            elif use_claude_vision:
                print("→ Diagrams will be processed ONLY by Claude Vision.")
            else:
                print("→ No Vision AI selected. Using OCR only (fastest).")

            try:
                t0 = time.time()
                engine.build_index(
                    pdf_path=pdf,
                    gantt_path=gantt,
                    class_path=cls,
                    seq_path=seq,
                    use_qwen_vision=use_qwen_vision,
                    use_claude_vision=use_claude_vision,
                )
                t1 = time.time()
                print(f"✔ Indexed successfully in {t1 - t0:.2f}s.")
            except Exception as e:
                print(f"[ERROR] Indexing failed: {e}")

        # ----- CHAT -----
        elif choice == "2":
            if not engine.vectorstore:
                print("Please index documents first (Option 1).")
                continue

            if claude_text_llm is None:
                try:
                    claude_text_llm = ClaudeAnswerer()
                    print("✔ Claude (text) initialized for final answers.")
                except Exception as e:
                    print(f"[ERROR] Failed to init Claude for answers: {e}")
                    print(
                        "Make sure 'anthropic' is installed and ANTHROPIC_API_KEY is set."
                    )
                    continue

            while True:
                q = input("\n[You]: ").strip()
                if q.lower() in ("exit", "back", "quit"):
                    break

                total_start = time.time()

                # Retrieval
                retrieval_start = time.time()
                try:
                    results = engine.hybrid_search(q, top_k=7)
                except Exception as e:
                    print(f"[ERROR] Search failed: {e}")
                    continue
                retrieval_time = time.time() - retrieval_start
                print(f"[Retrieved in {retrieval_time:.2f}s]")

                if not results:
                    print("No matching content found in the SRD or diagrams.")
                    continue

                for r in results:
                    src = r["metadata"].get("source")
                    sect = r["metadata"].get("section", "N/A")
                    score = r["score"]
                    print(f" → {src} | section={sect} | score={score:.2f}")

                print("\n--- Claude Answer ---")

                # Answer
                claude_start = time.time()
                try:
                    answer = claude_text_llm.generate_answer(q, results)
                    print(answer)
                except Exception as e:
                    print(f"Claude error: {e}")
                claude_time = time.time() - claude_start

                total_time = time.time() - total_start
                print("\n[Timings]")
                print(f" - Retrieval time: {retrieval_time:.2f}s")
                print(f" - Claude answer call time (wrapper): {claude_time:.2f}s")
                print(f" - Total time (Question → Final Answer): {total_time:.2f}s")
                print("---------------------")

        # ----- EXIT -----
        elif choice == "3":
            print("Goodbye.")
            break

        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
