# argparse를 위한 커스텀 Action 클래스
import argparse
import ast # ast 모듈 임포트

class DictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        options_dict = {}
        for kv_pair in values:
            key, value_str = kv_pair.split('=', 1)
            keys = key.split('.')
            
            current_level = options_dict
            for k in keys[:-1]:
                if k not in current_level:
                    current_level[k] = {}
                current_level = current_level[k]
                
            final_key = keys[-1]
            try:
                # --- 여기가 핵심 변경 사항 ---
                # eval 대신 ast.literal_eval 사용
                evaluated_value = ast.literal_eval(value_str)
                current_level[final_key] = evaluated_value
            except (ValueError, SyntaxError):
                # literal_eval은 변환 실패 시 ValueError 또는 SyntaxError 발생
                # 실패 시 그냥 원래 문자열 값을 사용
                current_level[final_key] = value_str

        setattr(namespace, self.dest, options_dict)
        
def recursive_update(d, u):
    """딕셔너리를 재귀적으로 업데이트합니다."""
    for k, v in u.items():
        if isinstance(v, dict):
            # d.get(k, {})는 d에 k가 없으면 빈 dict를 반환
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class Config(dict):
    """
    점을 사용하여 속성에 접근할 수 있는 커스텀 딕셔너리 클래스.
    예: config.model.type
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    @staticmethod
    def from_dict(d):
        """
        중첩된 딕셔너리를 Config 객체로 재귀적으로 변환합니다.
        """
        if not isinstance(d, dict):
            return d
        
        config = Config()
        for k, v in d.items():
            config[k] = Config.from_dict(v)
        
        return config