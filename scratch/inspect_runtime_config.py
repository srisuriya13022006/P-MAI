from app.llm.factory import get_llm_provider, create_cloud_provider
from app.llm.resilient_provider import ResilientLLMProvider
from app.llm.cascading_provider import CascadingCloudProvider

def inspect():
    llm = get_llm_provider()
    print('get_llm_provider type:', type(llm).__name__)
    if isinstance(llm, ResilientLLMProvider):
        print('primary_provider type:', type(llm.primary_provider).__name__)
        if isinstance(llm.primary_provider, CascadingCloudProvider):
            for idx, p in enumerate(llm.primary_provider.providers):
                print(f'  cascade [{idx}]: {p.provider_name} (model={getattr(p, "model", "unknown")})')
        print('fallback_provider type:', type(llm.fallback_provider).__name__)
        if llm.fallback_provider:
            print(f'  fallback: {llm.fallback_provider.provider_name} (model={getattr(llm.fallback_provider, "model", "unknown")}, num_ctx={getattr(llm.fallback_provider, "num_ctx", None)})')

if __name__ == '__main__':
    inspect()
