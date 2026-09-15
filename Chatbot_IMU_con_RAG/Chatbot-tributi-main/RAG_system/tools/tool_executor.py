import json

from RAG_system.retrieval.choose_vocabulary import choose_vocab
from RAG_system.utils.date_utils import get_day_of_week


class ToolExecutor:

    def __init__(self, rag_tools):
        self.rag_tools = rag_tools

    def execute(self, tool_name, k, args, args_used, vocabulary_used):

        args_frozen = json.dumps(args, sort_keys=True)

        if args_frozen in args_used:
            return {"sources": "Arguments already used"}, {"agent_sources": []}

        if tool_name == "retrieve_context":

            tool_result, agent_tool_results = self.rag_tools.retrieve_context(k=k,
                **args
            )

        elif tool_name == "retrieve_normativa":

            reference = args.get("reference", {})
            articolo = reference.get("articolo")
            comma = reference.get("comma")

            args_for_call = {k: v for k, v in args.items() if k != "reference"}

            tool_result, agent_tool_results = self.rag_tools.retrieve_normativa(
                articolo=articolo,
                comma=comma,
                **args_for_call
            )

        elif tool_name == "get_day_of_week":

            tool_result = get_day_of_week(**args)
            agent_tool_results = tool_result

        elif tool_name == "get_vocabulary":
            vocab_sources, agent_vocab, used_voc = choose_vocab(
                query=args["query"],
                rag_tools=self.rag_tools,
                vocabulary_used=vocabulary_used
            )

            agent_tool_results = {"agent_sources": agent_vocab, "used": used_voc}

            tool_result = vocab_sources

        else:
            raise RuntimeError(f"Tool non gestito: {tool_name}")

        return tool_result, agent_tool_results