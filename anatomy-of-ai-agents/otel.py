from uuid import UUID
from langchain_core.messages import BaseMessage
from typing import Any, Dict, List, Optional
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

class OTELCompliantLangfuseHandler(LangfuseCallbackHandler):
    """
    Custom Langfuse handler that follows OTEL semantic conventions.

    According to OTEL spec for GenAI:
    - gen_ai.tool.definitions should be in metadata (opt-in attribute)
    - Tool definitions should NOT be in the input/prompt

    This handler:
    1. Extracts tool definitions from invocation_params
    2. Adds them to metadata as 'schema_definitions'
    3. Removes them from invocation_params to prevent adding to input

    Handles both:
    - bind_tools() -> captures tool definitions
    - with_structured_output() -> captures Pydantic schema
    """

    def on_chat_model_start(
            self,
            serialized: Optional[Dict[str, Any]],
            messages: List[List[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        try:
            # Extract tool/schema definitions
            invocation_params = kwargs.get("invocation_params", {})
            tools = invocation_params.get("tools", [])
            format_param = invocation_params.get("format")

            # Prepare schema info for metadata
            schema_info = None
            if tools:
                schema_info = {"type": "tools", "definitions": tools}
            elif format_param:
                schema_info = {"type": "tools", "definitions": format_param}

            # Add to metadata and remove from input
            if schema_info:
                if metadata is None:
                    metadata = {}
                metadata["schema_definitions"] = schema_info

                # Remove tools from invocation_params to prevent adding to input
                kwargs_copy = kwargs.copy()
                invocation_params_copy = invocation_params.copy()
                invocation_params_copy.pop("tools", None)
                kwargs_copy["invocation_params"] = invocation_params_copy

                # Call parent with modified kwargs
                return super().on_chat_model_start(
                    serialized,
                    messages,
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    tags=tags,
                    metadata=metadata,
                    **kwargs_copy
                )

            # No tools/schema, call parent normally
            return super().on_chat_model_start(
                serialized,
                messages,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                **kwargs
            )

        except Exception as e:
            # Fallback to parent implementation on error
            return super().on_chat_model_start(
                serialized,
                messages,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                **kwargs
            )
