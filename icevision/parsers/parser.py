__all__ = ["ParserInterface", "Parser"]

from icevision.imports import *
from icevision.utils import *
from icevision.utils.code_template import *
from icevision.core import *
from icevision.data import *


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


# TODO: Rename to BaseParser
class ParserInterface(ABC):
    @abstractmethod
    def parse(
        self, data_splitter: DataSplitter, autofix: bool = True, show_pbar: bool = True
    ) -> List[List[RecordType]]:
        pass


class Parser(ParserInterface, ABC):
    """Base class for all parsers, implements the main parsing logic.

    The actual fields to be parsed are defined by the mixins used when
    defining a custom parser. The only required fields for all parsers
    are `image_id` and `image_width_height`.

    # Arguments
        idmap: Maps from filenames to unique ids, pass an `IDMap()` if you need this information.

    # Examples

    Create a parser for image filepaths.
    ```python
    class FilepathParser(Parser, FilepathParserMixin):
        # implement required abstract methods
    ```
    """

    def __init__(
        self,
        template_record,
        class_map: Optional[ClassMap] = None,
        idmap: Optional[IDMap] = None,
    ):
        # self.class_map = class_map or ClassMap()
        # if class_map is None:
        #     self.class_map.unlock()
        self.template_record = template_record
        self.idmap = idmap or IDMap()

    @abstractmethod
    def __iter__(self) -> Any:
        pass

    @abstractmethod
    def parse_fields(self, o, record: BaseRecord, is_new: bool) -> None:
        pass

    def create_record(self) -> BaseRecord:
        return deepcopy(self.template_record)

    def prepare(self, o):
        pass

    def parse_dicted(self, show_pbar: bool = True) -> Dict[int, RecordType]:
        records = RecordCollection(self.create_record)

        for sample in pbar(self, show_pbar):
            try:
                self.prepare(sample)
                record_id = self.record_id(sample)
                record = records.get_by_record_id(record_id)
                self.parse_fields(sample, record=record, is_new=record.is_new)

            except AbortParseRecord as e:
                logger.warning(
                    "Record with record_id: {} was skipped because: {}",
                    record_id,
                    str(e),
                )

        return records

    def _check_path(self, path: Union[str, Path] = None):
        if path is None:
            return False
        if path is not None:
            return Path(path).exists()

    def parse(
        self,
        data_splitter: DataSplitter = None,
        autofix: bool = True,
        show_pbar: bool = True,
        cache_filepath: Union[str, Path] = None,
    ) -> List[List[BaseRecord]]:
        """Loops through all data points parsing the required fields.

        # Arguments
            data_splitter: How to split the parsed data, defaults to a [0.8, 0.2] random split.
            show_pbar: Whether or not to show a progress bar while parsing the data.
            cache_filepath: Path to save records in pickle format. Defaults to None, e.g.
                            if the user does not specify a path, no saving nor loading happens.

        # Returns
            A list of records for each split defined by `data_splitter`.
        """
        if self._check_path(cache_filepath):
            logger.info(
                f"Loading cached records from {cache_filepath}",
            )
            return pickle.load(open(Path(cache_filepath), "rb"))
        else:
            data_splitter = data_splitter or RandomSplitter([0.8, 0.2])
            records = self.parse_dicted(show_pbar=show_pbar)

            if autofix:
                logger.opt(colors=True).info("<blue><bold>Autofixing records</></>")
                records = records.autofix()

            records = records.make_splits(data_splitter)

            # self.class_map.lock()
            if cache_filepath is not None:
                pickle.dump(records, open(Path(cache_filepath), "wb"))

            return records

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return ["def __iter__(self) -> Any:"] + templates

    @classmethod
    def generate_template(cls, record):
        record_builder_template = record.builder_template()

        template = CodeTemplate()
        template.add_line(f"class MyParser({cls.__name__}):", 0)
        template.add_line(f"def __init__(self, template_record):", 1)
        template.add_line(f"super().__init__(template_record=template_record)", 2)
        template.add_line(f"def __iter__(self) -> Any:", 1)
        template.add_line(f"def __len__(self) -> int:", 1)
        # template.add_line("def create_record(self) -> BaseRecord:", 1)
        # template.add_line(f"return {record.__class__.__name__}({components_names})", 2)
        template.add_line("def record_id(self, o: Any) -> Hashable:", 1)
        template.add_line(
            "def parse_fields(self, o: Any, record: BaseRecord, is_new: bool):", 1
        )
        template.add_lines(record_builder_template, 2)

        template.display()
