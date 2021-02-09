package ai.h2o.automl.leaderboard;

import hex.Model;
import water.Iced;
import water.Key;
import water.util.Log;

public class AlgoName extends Iced<AlgoName> implements LeaderboardCell<String, AlgoName> {
    public static final LeaderboardColumn COLUMN = new LeaderboardColumn("algo", "string", "%s");
    final Key<Model> _modelId;
    private String _algo;

    public AlgoName(Model model) {
        this._modelId = model._key;
        setValue(model._parms.algoName());
    }

    @Override
    public LeaderboardColumn getColumn() {
        return COLUMN;
    }

    @Override
    public Key<Model> getModelId() {
        return _modelId;
    }

    @Override
    public String getValue() {
        return _algo;
    }

    @Override
    public void setValue(String value) {
        _algo = value;
    }

    @Override
    public boolean isNA() {
        return false;
    }

    @Override
    public String fetch() {
        if (getValue() == null) {
            try {
                Model model = _modelId.get();
                setValue(model._parms.algoName());
            } catch (Exception e) {
                Log.err("Could not retrieve algoName for model "+_modelId, e);
                setValue("UNKNOWN");
            }
        }
        return getValue();
    }
}